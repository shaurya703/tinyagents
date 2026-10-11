//! Policy inputs and observations for after-model stop hooks.
use async_trait::async_trait;
use tinyinference_llm::model::ModelResponse;

/// Host-defined completed-call accounting used by stop policies.
///
/// The harness owns when a completed response is recorded. The host owns its
/// metrics, pricing source and treatment of unreported usage or charges.
pub trait StopHookMetrics: Clone + Default + Send + Sync + 'static {
    /// Fold this completed response into the current run's tally.
    fn record_model(&mut self, model: &str, response: &ModelResponse);
}

/// Hooks without a cost or usage policy can use an empty tally.
impl StopHookMetrics for () {
    fn record_model(&mut self, _model: &str, _response: &ModelResponse) {}
}

/// Snapshot delivered after one completed model call.
///
/// Every hook observes the same borrowed tally, including hooks registered
/// after a policy that votes to stop. Clone values that must outlive the check.
pub struct TurnState<'a, Metrics = ()> {
    /// One-based count of model responses observed in this run.
    pub iteration: u32,
    /// Live model-call cap, saturated to the counter's range.
    pub max_iterations: u32,
    /// Host-defined cumulative accounting through the completed call.
    pub cost: &'a Metrics,
    /// Host-supplied model label for accounting and diagnostics.
    pub model: &'a str,
}

/// A policy or observer evaluated between model calls.
#[async_trait]
pub trait StopHook<Metrics = ()>: Send + Sync {
    /// Stable policy name used in the first stopping report.
    fn name(&self) -> &str;
    /// Observe the completed call and decide whether another call is permitted.
    async fn check(&self, state: &TurnState<'_, Metrics>) -> StopDecision;
}

/// Vote from one policy; all policies still observe the completed call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopDecision {
    /// Continue the current run.
    Continue,
    /// Pause before the next model call with this explanation.
    Stop {
        /// Human-readable reason supplied by the policy.
        reason: String,
    },
}

/// The first stopping policy, delivered once to the host's result projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopReport {
    /// Name of the first policy that voted to stop.
    pub hook: String,
    /// Explanation supplied by that policy.
    pub reason: String,
    /// Completed model calls observed when the policy stopped.
    pub iteration: u32,
    /// Host-supplied model label.
    pub model: String,
}

/// State belongs to a context instance, never its reusable durable run id.
#[derive(Default)]
pub(super) struct HookRunState<Metrics> {
    pub(super) lifecycle: std::sync::Weak<()>,
    pub(super) metrics: Metrics,
    pub(super) iteration: u32,
    pub(super) stopped: bool,
    pub(super) pause_queued: bool,
}
