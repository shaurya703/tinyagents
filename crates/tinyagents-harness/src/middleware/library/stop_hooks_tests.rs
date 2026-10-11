use super::*;
use crate::steering::{SteeringCommand, SteeringHandle, SteeringTarget};

#[derive(Clone, Default)]
struct Metrics {
    completed_calls: usize,
}

impl StopHookMetrics for Metrics {
    fn record_model(&mut self, model: &str, _response: &ModelResponse) {
        self.completed_calls += 1;
        assert_eq!(model, "model");
    }
}

#[derive(Clone, Default)]
struct ModelMetrics {
    models: Vec<String>,
}

impl StopHookMetrics for ModelMetrics {
    fn record_model(&mut self, model: &str, _response: &ModelResponse) {
        self.models.push(model.to_owned());
    }
}

struct ModelHook;

#[async_trait]
impl StopHook<ModelMetrics> for ModelHook {
    fn name(&self) -> &str {
        "observer"
    }

    async fn check(&self, state: &TurnState<'_, ModelMetrics>) -> StopDecision {
        assert_eq!(state.model, "fallback");
        assert_eq!(state.cost.models, ["fallback"]);
        StopDecision::Continue
    }
}

type Observations = Arc<Mutex<Vec<(&'static str, u32, usize)>>>;

struct Hook {
    name: &'static str,
    stop: bool,
    observations: Observations,
}

#[async_trait]
impl StopHook<Metrics> for Hook {
    fn name(&self) -> &str {
        self.name
    }

    async fn check(&self, state: &TurnState<'_, Metrics>) -> StopDecision {
        self.observations.lock().unwrap().push((
            self.name,
            state.iteration,
            state.cost.completed_calls,
        ));
        assert_eq!(state.max_iterations, 5);
        assert_eq!(state.model, "model");
        if self.stop {
            StopDecision::Stop {
                reason: format!("{} voted stop", self.name),
            }
        } else {
            StopDecision::Continue
        }
    }
}

/// Each run receives its own pause queue, even when middleware is shared.
fn run() -> RunContext {
    RunContext::new(RunConfig::new("same-run-id").with_max_model_calls(5), ())
        .with_steering(SteeringHandle::allow_all())
}

/// A later observer still accounts for the call that made an earlier policy stop.
#[tokio::test]
async fn every_stop_hook_observes_the_stopping_call_and_the_first_reason_pauses_once() {
    let observations = Arc::new(Mutex::new(Vec::new()));
    let reports = Arc::new(Mutex::new(Vec::new()));
    let report_sink = reports.clone();
    let middleware = StopHookMiddleware::<Metrics>::new(
        "model",
        vec![
            Arc::new(Hook {
                name: "first",
                stop: true,
                observations: observations.clone(),
            }),
            Arc::new(Hook {
                name: "second",
                stop: true,
                observations: observations.clone(),
            }),
            Arc::new(Hook {
                name: "observer",
                stop: false,
                observations: observations.clone(),
            }),
        ],
    )
    .with_stop_observer(move |report| report_sink.lock().unwrap().push(report.clone()));
    let mut context = run();
    let mut response = ModelResponse::assistant("done");
    middleware
        .after_model(&mut context, &(), &mut response)
        .await
        .unwrap();
    middleware
        .after_model(&mut context, &(), &mut response)
        .await
        .unwrap();

    assert_eq!(
        *observations.lock().unwrap(),
        vec![("first", 1, 1), ("second", 1, 1), ("observer", 1, 1),]
    );
    let reports = reports.lock().unwrap();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].hook, "first");
    assert_eq!(reports[0].reason, "first voted stop");
    assert_eq!(reports[0].iteration, 1);
    assert_eq!(
        context
            .steering
            .as_ref()
            .unwrap()
            .drain()
            .iter()
            .filter(|command| matches!(command, SteeringCommand::Pause))
            .count(),
        1
    );
}

/// Equal durable ids do not share counters, metrics or pause latches.
#[tokio::test]
async fn stop_hook_state_is_isolated_between_context_instances() {
    let observations = Arc::new(Mutex::new(Vec::new()));
    let middleware = StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "budget",
            stop: true,
            observations: observations.clone(),
        })],
    );
    let mut first = run();
    let mut second = run();
    assert_ne!(first.instance_id(), second.instance_id());
    for context in [&mut first, &mut second] {
        middleware
            .after_model(context, &(), &mut ModelResponse::assistant("done"))
            .await
            .unwrap();
        assert_eq!(context.steering.as_ref().unwrap().drain().len(), 1);
    }
    assert_eq!(
        *observations.lock().unwrap(),
        vec![("budget", 1, 1), ("budget", 1, 1)]
    );
}

/// Error cleanup retires a partial tally before the same context is reused.
#[tokio::test]
async fn stop_hook_error_cleanup_restarts_the_completed_call_tally() {
    let observations = Arc::new(Mutex::new(Vec::new()));
    let middleware = StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "observer",
            stop: false,
            observations: observations.clone(),
        })],
    );
    let mut context = run();
    middleware
        .after_model(&mut context, &(), &mut ModelResponse::assistant("first"))
        .await
        .unwrap();
    <StopHookMiddleware<Metrics> as Middleware<(), ()>>::on_error(
        &middleware,
        &mut context,
        &TinyAgentsError::Model("failed".into()),
    )
    .await
    .unwrap();
    middleware
        .after_model(&mut context, &(), &mut ModelResponse::assistant("second"))
        .await
        .unwrap();
    assert_eq!(
        *observations.lock().unwrap(),
        vec![("observer", 1, 1), ("observer", 1, 1)]
    );
}

/// An error after a stop vote retracts the queued pause before handle reuse.
#[tokio::test]
async fn stop_hook_error_cleanup_retracts_a_queued_pause() {
    let middleware = StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "budget",
            stop: true,
            observations: Arc::new(Mutex::new(Vec::new())),
        })],
    );
    let mut context = run();
    middleware
        .after_model(&mut context, &(), &mut ModelResponse::assistant("first"))
        .await
        .unwrap();
    assert!(!context.steering.as_ref().unwrap().is_empty());
    <StopHookMiddleware<Metrics> as Middleware<(), ()>>::on_error(
        &middleware,
        &mut context,
        &TinyAgentsError::Model("failed after response".into()),
    )
    .await
    .unwrap();
    assert!(context.steering.as_ref().unwrap().is_empty());
}

/// An unrelated pause queued by the caller survives an error when no hook
/// voted to stop.
#[tokio::test]
async fn stop_hook_error_cleanup_preserves_a_caller_pause() {
    let middleware = StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "observer",
            stop: false,
            observations: Arc::new(Mutex::new(Vec::new())),
        })],
    );
    let mut context = run();
    context.steering.as_ref().unwrap().send_to(
        SteeringTarget::Run(context.run_id().clone()),
        SteeringCommand::Pause,
    );

    <StopHookMiddleware<Metrics> as Middleware<(), ()>>::on_error(
        &middleware,
        &mut context,
        &TinyAgentsError::Model("failed before the hook could stop".into()),
    )
    .await
    .unwrap();

    assert_eq!(
        context.steering.as_ref().unwrap().drain(),
        vec![SteeringCommand::Pause]
    );
}

/// When both producers queue the same pause, cleanup retracts the hook's
/// later entry and leaves the caller's command intact.
#[tokio::test]
async fn stop_hook_error_cleanup_retracts_only_its_pause() {
    let middleware = StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "budget",
            stop: true,
            observations: Arc::new(Mutex::new(Vec::new())),
        })],
    );
    let mut context = run();
    let steering = context.steering.as_ref().unwrap().clone();
    steering.send_to(
        SteeringTarget::Run(context.run_id().clone()),
        SteeringCommand::Pause,
    );
    middleware
        .after_model(&mut context, &(), &mut ModelResponse::assistant("first"))
        .await
        .unwrap();

    <StopHookMiddleware<Metrics> as Middleware<(), ()>>::on_error(
        &middleware,
        &mut context,
        &TinyAgentsError::Model("failed after stop".into()),
    )
    .await
    .unwrap();

    assert_eq!(
        context.steering.as_ref().unwrap().drain(),
        vec![SteeringCommand::Pause]
    );
}

/// Completed-call accounting uses the runtime-selected model when available.
#[tokio::test]
async fn stop_hook_metrics_receive_the_resolved_model_name() {
    let middleware = StopHookMiddleware::<ModelMetrics>::new("primary", vec![Arc::new(ModelHook)]);
    let mut context = run();
    let mut response = ModelResponse::assistant("fallback");
    response.resolved_model = Some(tinyinference_llm::model::ResolvedModel {
        name: "fallback".into(),
        requested: Some("primary".into()),
        source: tinyinference_llm::model::ModelResolutionSource::RequestOverride,
    });
    middleware
        .after_model(&mut context, &(), &mut response)
        .await
        .unwrap();
}

/// A stopping policy must not silently continue when its pause cannot be delivered.
#[tokio::test]
async fn stop_hooks_require_a_run_steering_handle_before_observing_calls() {
    let observations = Arc::new(Mutex::new(Vec::new()));
    let middleware = StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "budget",
            stop: true,
            observations: observations.clone(),
        })],
    );
    let mut context = RunContext::new(RunConfig::new("unsteered"), ());
    assert!(middleware.before_agent(&mut context, &()).await.is_err());
    let error = middleware
        .after_model(&mut context, &(), &mut ModelResponse::assistant("done"))
        .await
        .unwrap_err();
    assert!(matches!(error, TinyAgentsError::Middleware(_)));
    assert!(observations.lock().unwrap().is_empty());
}

/// Successful cleanup also makes a reused context begin with an empty tally.
#[tokio::test]
async fn stop_hook_success_cleanup_restarts_the_completed_call_tally() {
    let observations = Arc::new(Mutex::new(Vec::new()));
    let middleware = StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "observer",
            stop: false,
            observations: observations.clone(),
        })],
    );
    let mut context = run();
    middleware
        .after_model(&mut context, &(), &mut ModelResponse::assistant("first"))
        .await
        .unwrap();
    middleware
        .after_agent(&mut context, &(), &mut crate::middleware::AgentRun::new())
        .await
        .unwrap();
    middleware
        .after_model(&mut context, &(), &mut ModelResponse::assistant("second"))
        .await
        .unwrap();
    assert_eq!(
        *observations.lock().unwrap(),
        vec![("observer", 1, 1), ("observer", 1, 1)]
    );
}

/// A harmless read tool gives the model a reason to request a second call.
struct ReadTool;

#[async_trait]
impl tinytools::Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }
    fn description(&self) -> &str {
        "Read fixed test data"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    fn policy(&self) -> tinytools::ToolPolicy {
        tinytools::ToolPolicy::read_only()
    }
    async fn execute(&self, _: serde_json::Value) -> anyhow::Result<tinytools::ToolResult> {
        Ok(tinytools::ToolResult::success("fixed result"))
    }
}

/// The hook stops the real loop before a second provider call can be dispatched.
#[tokio::test]
async fn stop_hook_pause_prevents_the_next_model_call_in_the_native_loop() {
    let observations = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(tinyinference_llm::providers::MockModel::with_tool_call(
        "read",
        serde_json::json!({}),
    ));
    let mut harness: crate::runtime::AgentHarness<()> = crate::runtime::AgentHarness::new();
    harness.register_model("model", model.clone());
    harness.register_tool(Arc::new(ReadTool));
    harness.push_middleware(Arc::new(StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "budget",
            stop: true,
            observations: observations.clone(),
        })],
    )));
    let outcome = harness
        .invoke_in_context_with_status(&(), run(), vec![Message::user("read and continue")])
        .await
        .unwrap();
    assert_eq!(model.call_count(), 1);
    assert_eq!(outcome.run.model_calls, 1);
    assert_eq!(
        outcome.run.tool_calls, 1,
        "the completed call keeps its pending tool round"
    );
    assert_eq!(*observations.lock().unwrap(), vec![("budget", 1, 1)]);
}

/// No policies means no steering requirement and no fabricated accounting.
#[tokio::test]
async fn an_empty_stop_hook_list_leaves_unsteered_calls_unchanged() {
    let middleware = StopHookMiddleware::<()>::new("model", Vec::new());
    let mut context = RunContext::new(RunConfig::new("unsteered"), ());
    let mut response = ModelResponse::assistant("unchanged");
    middleware.before_agent(&mut context, &()).await.unwrap();
    middleware
        .after_model(&mut context, &(), &mut response)
        .await
        .unwrap();
    assert_eq!(response.text(), "unchanged");
}

#[derive(Clone, Default)]
struct LifetimeMetrics(Arc<()>);

impl StopHookMetrics for LifetimeMetrics {
    fn record_model(&mut self, _: &str, _: &ModelResponse) {}
}

struct LifetimeObserver(Arc<Mutex<Vec<std::sync::Weak<()>>>>);

#[async_trait]
impl StopHook<LifetimeMetrics> for LifetimeObserver {
    fn name(&self) -> &str {
        "lifetime"
    }
    async fn check(&self, state: &TurnState<'_, LifetimeMetrics>) -> StopDecision {
        self.0.lock().unwrap().push(Arc::downgrade(&state.cost.0));
        StopDecision::Continue
    }
}

/// Dropping a run without a terminal callback cannot leak its accounting indefinitely.
#[tokio::test]
async fn stop_hook_reclaims_abandoned_context_accounting_on_the_next_call() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let middleware = StopHookMiddleware::<LifetimeMetrics>::new(
        "model",
        vec![Arc::new(LifetimeObserver(seen.clone()))],
    );
    let mut first = run();
    middleware
        .after_model(&mut first, &(), &mut ModelResponse::assistant("first"))
        .await
        .unwrap();
    drop(first);
    let mut next = run();
    middleware
        .after_model(&mut next, &(), &mut ModelResponse::assistant("next"))
        .await
        .unwrap();
    let seen = seen.lock().unwrap();
    assert!(
        seen[0].upgrade().is_none(),
        "abandoned accounting is retired"
    );
    assert!(
        seen[1].upgrade().is_some(),
        "the current run keeps its tally"
    );
}

struct CapObserver(Arc<Mutex<Vec<u32>>>);

#[async_trait]
impl StopHook for CapObserver {
    fn name(&self) -> &str {
        "cap"
    }
    async fn check(&self, state: &TurnState<'_>) -> StopDecision {
        self.0.lock().unwrap().push(state.max_iterations);
        StopDecision::Continue
    }
}

/// A narrowed live policy governs admission even when RunConfig requested more calls.
#[tokio::test]
async fn stop_hook_reports_the_live_model_call_cap_after_policy_narrowing() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let middleware =
        StopHookMiddleware::<()>::new("model", vec![Arc::new(CapObserver(seen.clone()))]);
    let mut context = run();
    context.limits = crate::limits::LimitTracker::new(
        crate::limits::RunLimits::default().with_max_model_calls(2),
    );
    middleware
        .after_model(&mut context, &(), &mut ModelResponse::assistant("done"))
        .await
        .unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![2]);
}

/// A steering queue that forbids pause cannot silently bypass a stop policy.
#[tokio::test]
async fn stop_hooks_reject_a_steering_policy_that_disallows_pause() {
    let observations = Arc::new(Mutex::new(Vec::new()));
    let middleware = StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "budget",
            stop: true,
            observations: observations.clone(),
        })],
    );
    let mut context =
        run().with_steering(SteeringHandle::new(crate::steering::SteeringPolicy::new()));
    let result = middleware.before_agent(&mut context, &()).await;
    assert!(matches!(result, Err(TinyAgentsError::Middleware(_))));
    let result = middleware
        .after_model(&mut context, &(), &mut ModelResponse::assistant("done"))
        .await;
    assert!(matches!(result, Err(TinyAgentsError::Middleware(_))));
    assert!(observations.lock().unwrap().is_empty());
}

/// A child stop targets its own steering lane and never leaks a pause to its parent.
#[tokio::test]
async fn stop_hook_on_a_child_run_pauses_only_that_child_before_another_call() {
    let parent = run();
    let parent_handle = parent.steering.clone().unwrap();
    let child = parent
        .child(RunConfig::new("child").with_max_model_calls(5), ())
        .unwrap();
    let child_handle = child.steering.clone().unwrap();
    let model = Arc::new(tinyinference_llm::providers::MockModel::with_tool_call(
        "read",
        serde_json::json!({}),
    ));
    let mut harness: crate::runtime::AgentHarness<()> = crate::runtime::AgentHarness::new();
    harness.register_model("model", model.clone());
    harness.register_tool(Arc::new(ReadTool));
    harness.push_middleware(Arc::new(StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "budget",
            stop: true,
            observations: Arc::new(Mutex::new(Vec::new())),
        })],
    )));
    let outcome = harness
        .invoke_in_context_with_status(&(), child, vec![Message::user("read and continue")])
        .await
        .unwrap();
    assert_eq!(model.call_count(), 1);
    assert_eq!(outcome.run.tool_calls, 1);
    assert!(child_handle.is_paused());
    assert!(!parent_handle.is_paused());
    assert!(
        parent_handle.is_empty(),
        "no child stop command remains for the parent"
    );
}

/// A stop on a text-only completion is consumed before natural finish and handle reuse.
#[tokio::test]
async fn stop_hook_on_a_final_response_reports_pause_without_a_leftover_command() {
    let context = run();
    let handle = context.steering.clone().unwrap();
    let mut harness: crate::runtime::AgentHarness<()> = crate::runtime::AgentHarness::new();
    harness.register_model(
        "model",
        Arc::new(tinyinference_llm::providers::MockModel::constant("done")),
    );
    harness.push_middleware(Arc::new(StopHookMiddleware::<Metrics>::new(
        "model",
        vec![Arc::new(Hook {
            name: "budget",
            stop: true,
            observations: Arc::new(Mutex::new(Vec::new())),
        })],
    )));
    let outcome = harness
        .invoke_in_context_with_status(&(), context, vec![Message::user("answer")])
        .await
        .unwrap();
    assert_eq!(
        outcome.run.terminal.as_ref().unwrap().reason,
        crate::terminal::TerminalReason::Paused
    );
    assert!(handle.is_paused());
    assert!(handle.is_empty());
    handle.resume();
    let mut next: crate::runtime::AgentHarness<()> = crate::runtime::AgentHarness::new();
    next.register_model(
        "model",
        Arc::new(tinyinference_llm::providers::MockModel::constant("next")),
    );
    let outcome = next
        .invoke_in_context_with_status(
            &(),
            run().with_steering(handle.clone()),
            vec![Message::user("next")],
        )
        .await
        .unwrap();
    assert_eq!(
        outcome.run.terminal.as_ref().unwrap().class,
        crate::terminal::TerminalClass::Success
    );
    assert!(!handle.is_paused());
    assert!(handle.is_empty());
}
