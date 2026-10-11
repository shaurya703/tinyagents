//! Stop-policy execution after completed model calls.
//!
//! The host supplies policy hooks, accounting and stop-result presentation.
//! This middleware owns ordering, per-run tallies and once-only pause delivery.
mod types;
pub use types::{StopDecision, StopHook, StopHookMetrics, StopReport, TurnState};

use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::middleware::{AgentRun, Middleware};
use crate::steering::{SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringTarget};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tinyinference_llm::model::ModelResponse;
use types::HookRunState;

type StopObserver = Arc<dyn Fn(&StopReport) + Send + Sync>;

/// Evaluate every installed policy after each completed model call.
///
/// The first stopping reason wins, but later hooks still observe that call.
/// A pause is sent once on the current context's steering handle; its policy
/// must allow pauses. Already-completed tool calls from that response retain
/// their normal execution path. A reused middleware isolates counters, metrics
/// and stop latches by context instance, with cleanup on success and error.
pub struct StopHookMiddleware<Metrics = ()> {
    model: String,
    hooks: Vec<Arc<dyn StopHook<Metrics>>>,
    runs: Mutex<HashMap<u64, HookRunState<Metrics>>>,
    observer: Option<StopObserver>,
}

impl<Metrics: StopHookMetrics> StopHookMiddleware<Metrics> {
    /// Build the shared mechanics over host-provided policies and model label.
    ///
    /// A nonempty hook list requires a steering handle on every run context.
    pub fn new(model: impl Into<String>, hooks: Vec<Arc<dyn StopHook<Metrics>>>) -> Self {
        Self {
            model: model.into(),
            hooks,
            runs: Mutex::new(HashMap::new()),
            observer: None,
        }
    }

    /// Project the first stopping report into a host result, log or event.
    ///
    /// Called once, after all policies observe the completed call and before
    /// the pause is queued. The callback does not hold the accounting mutex.
    pub fn with_stop_observer<F>(mut self, observer: F) -> Self
    where
        F: Fn(&StopReport) + Send + Sync + 'static,
    {
        self.observer = Some(Arc::new(observer));
        self
    }

    /// Refuse silent policy bypass when no pause queue was installed.
    fn steering<Ctx>(&self, context: &RunContext<Ctx>) -> Result<SteeringHandle> {
        let handle = context.steering.clone().ok_or_else(|| {
            TinyAgentsError::Middleware(
                "stop hooks require a steering handle on the run context".into(),
            )
        })?;
        if !handle.policy().is_allowed(SteeringCommandKind::Pause) {
            return Err(TinyAgentsError::Middleware(
                "stop hooks require a steering policy that permits pause".into(),
            ));
        }
        Ok(handle)
    }

    /// Retire a terminal run's accounting and stop latch.
    fn forget<Ctx>(&self, context: &RunContext<Ctx>) {
        self.runs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&context.instance_id());
    }
}

#[async_trait]
impl<State, Ctx, Metrics> Middleware<State, Ctx> for StopHookMiddleware<Metrics>
where
    State: Send + Sync,
    Ctx: Send + Sync,
    Metrics: StopHookMetrics,
{
    fn name(&self) -> &str {
        "stop_hooks"
    }

    async fn before_agent(&self, context: &mut RunContext<Ctx>, _state: &State) -> Result<()> {
        if !self.hooks.is_empty() {
            self.steering(context)?;
            let mut runs = self
                .runs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            runs.retain(|_, run| run.lifecycle.strong_count() > 0);
            runs.insert(
                context.instance_id(),
                HookRunState {
                    lifecycle: Arc::downgrade(&context.lifecycle),
                    ..HookRunState::default()
                },
            );
        }
        Ok(())
    }

    async fn after_model(
        &self,
        context: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        if self.hooks.is_empty() {
            return Ok(());
        }
        let steering = self.steering(context)?;
        let model = response
            .resolved_model
            .as_ref()
            .map_or(self.model.as_str(), |resolved| resolved.name.as_str());
        let (iteration, metrics) = {
            let mut runs = self
                .runs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // A dropped context may bypass terminal hooks. Reclaim its tally
            // when admitting a new context without scanning on every call.
            if !runs.contains_key(&context.instance_id()) {
                runs.retain(|_, run| run.lifecycle.strong_count() > 0);
            }
            let run = runs
                .entry(context.instance_id())
                .or_insert_with(|| HookRunState {
                    lifecycle: Arc::downgrade(&context.lifecycle),
                    ..HookRunState::default()
                });
            if run.stopped {
                return Ok(());
            }
            run.iteration = run.iteration.saturating_add(1);
            run.metrics.record_model(model, response);
            (run.iteration, run.metrics.clone())
        };
        let state = TurnState {
            iteration,
            max_iterations: context
                .limits
                .limits()
                .max_model_calls
                .min(u32::MAX as usize) as u32,
            cost: &metrics,
            model,
        };
        let mut first_stop = None;
        for hook in &self.hooks {
            if let StopDecision::Stop { reason } = hook.check(&state).await
                && first_stop.is_none()
            {
                first_stop = Some(StopReport {
                    hook: hook.name().to_owned(),
                    reason,
                    iteration,
                    model: model.to_owned(),
                });
            }
        }
        if let Some(report) = first_stop {
            let mut runs = self
                .runs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let run = runs
                .get_mut(&context.instance_id())
                .expect("active stop-hook run");
            run.stopped = true;
            run.pause_queued = true;
            drop(runs);
            if let Some(observer) = &self.observer {
                observer(&report);
            }
            steering.send_to(
                SteeringTarget::Run(context.run_id().clone()),
                SteeringCommand::Pause,
            );
        }
        Ok(())
    }

    async fn after_agent(
        &self,
        context: &mut RunContext<Ctx>,
        _state: &State,
        _run: &mut AgentRun,
    ) -> Result<()> {
        self.forget(context);
        Ok(())
    }

    async fn on_error(
        &self,
        context: &mut RunContext<Ctx>,
        _error: &TinyAgentsError,
    ) -> Result<()> {
        let pause_queued = self
            .runs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&context.instance_id())
            .is_some_and(|run| run.pause_queued);
        if pause_queued && let Some(steering) = &context.steering {
            steering.retract_last(
                &SteeringTarget::Run(context.run_id().clone()),
                &SteeringCommand::Pause,
            );
        }
        self.forget(context);
        Ok(())
    }
}
