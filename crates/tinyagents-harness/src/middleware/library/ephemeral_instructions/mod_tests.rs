use super::*;
use crate::context::{RunConfig, RunContext};
use crate::error::TinyAgentsError;
use crate::middleware::{AgentRun, Middleware};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ModelProfile, ModelRequest};

fn context() -> RunContext {
    RunContext::new(RunConfig::new("run"), ())
}
fn request() -> ModelRequest {
    ModelRequest {
        messages: vec![
            Message::system("persona"),
            Message::user("read"),
            Message::tool("call", "original"),
        ],
        ..Default::default()
    }
}

/// Delivery occurs once in FIFO order on an ephemeral request copy.
#[tokio::test]
async fn ephemeral_instructions_are_delivered_once_without_changing_the_source_history() {
    let mut context = context();
    let queue = EphemeralInstructionQueue::new();
    queue.push(&context, "first");
    queue.push(&context, "second");
    let injector = queue.middleware();
    assert!(<EphemeralInstructionMiddleware as Middleware<(), ()>>::is_observer(&injector));
    let original = request();
    let mut outgoing = original.clone();
    injector
        .before_model_dispatch(&mut context, &(), &mut outgoing)
        .await
        .unwrap();
    assert_eq!(original.messages.len(), 3);
    assert_eq!(outgoing.messages[3].text(), "first");
    assert_eq!(outgoing.messages[4].text(), "second");
    let mut next = original.clone();
    injector
        .before_model_dispatch(&mut context, &(), &mut next)
        .await
        .unwrap();
    assert_eq!(next.messages, original.messages);
}

/// Hoisting providers keep the prompt head unchanged and receive a tail instruction.
#[tokio::test]
async fn ephemeral_instructions_for_hoisting_models_keep_the_system_prefix_unchanged() {
    let mut context = context();
    let queue = EphemeralInstructionQueue::new();
    queue.push(&context, "try a bounded read");
    let injector = queue.middleware();
    context.model_profile = Some(ModelProfile {
        hoists_system_messages: true,
        mid_conversation_system_messages: true,
        ..Default::default()
    });
    let mut outgoing = request();
    let prefix = outgoing.messages[0].clone();
    injector
        .before_model_dispatch(&mut context, &(), &mut outgoing)
        .await
        .unwrap();
    assert_eq!(outgoing.messages.len(), 3);
    assert_eq!(outgoing.messages[0], prefix);
    let tail = outgoing.messages.last().unwrap().text();
    assert!(tail.starts_with("original"));
    assert!(tail.contains("try a bounded read"));
}

/// A run ending before another request does not carry pending corrections forward.
#[tokio::test]
async fn ephemeral_instructions_are_cleared_after_success_and_error() {
    let mut context = context();
    let queue = EphemeralInstructionQueue::new();
    let injector = queue.middleware();
    queue.push(&context, "not sent");
    injector
        .after_agent(&mut context, &(), &mut AgentRun::default())
        .await
        .unwrap();
    assert!(queue.drain_all().is_empty());
    queue.push(&context, "also not sent");
    <EphemeralInstructionMiddleware as Middleware<(), ()>>::on_error(
        &injector,
        &mut context,
        &TinyAgentsError::Validation("failed".into()),
    )
    .await
    .unwrap();
    assert!(queue.drain_all().is_empty());
}

/// Concurrent contexts with the same durable id never consume each other's corrections.
#[tokio::test]
async fn ephemeral_instructions_are_isolated_by_context_instance() {
    let queue = EphemeralInstructionQueue::new();
    let mut first = context();
    let mut second = context();
    queue.push(&first, "first run");
    queue.push(&second, "second run");
    let injector = queue.middleware();
    let mut outgoing = request();
    injector
        .before_model_dispatch(&mut first, &(), &mut outgoing)
        .await
        .unwrap();
    assert_eq!(outgoing.messages.last().unwrap().text(), "first run");
    assert_eq!(queue.drain_for(&second), vec!["second run"]);
    queue.push(&first, "still first");
    <EphemeralInstructionMiddleware as Middleware<(), ()>>::on_error(
        &injector,
        &mut second,
        &TinyAgentsError::Validation("failed".into()),
    )
    .await
    .unwrap();
    assert_eq!(queue.drain_for(&first), vec!["still first"]);
}

/// Dropped invocations leave no correction for a later context admission.
#[tokio::test]
async fn ephemeral_instructions_reclaim_abandoned_contexts_on_admission() {
    let queue = EphemeralInstructionQueue::new();
    {
        let abandoned = context();
        queue.push(&abandoned, "stale");
    }
    let mut live = context();
    queue.push(&live, "live");
    queue
        .middleware()
        .before_agent(&mut live, &())
        .await
        .unwrap();
    assert_eq!(queue.drain_all(), vec!["live"]);
}

struct NativeProducer(EphemeralInstructionQueue);
#[async_trait::async_trait]
impl Middleware<(), ()> for NativeProducer {
    fn name(&self) -> &str {
        "instruction_producer"
    }
    async fn after_tool(
        &self,
        context: &mut RunContext,
        _: &(),
        _: &crate::middleware::ToolInvocationIdentity,
        _: &mut tinytools::ToolResult,
    ) -> crate::error::Result<()> {
        self.0.push(context, "bounded correction");
        Ok(())
    }
}
struct RequestCapture(std::sync::Arc<std::sync::Mutex<Vec<ModelRequest>>>);
#[async_trait::async_trait]
impl Middleware<(), ()> for RequestCapture {
    fn name(&self) -> &str {
        "request_capture"
    }
    async fn before_model_dispatch(
        &self,
        _: &mut RunContext,
        _: &(),
        request: &mut ModelRequest,
    ) -> crate::error::Result<()> {
        self.0.lock().unwrap().push(request.clone());
        Ok(())
    }
}
struct ReadTool;
#[async_trait::async_trait]
impl tinytools::Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }
    fn description(&self) -> &str {
        "test read"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type":"object"})
    }
    async fn execute(&self, _: serde_json::Value) -> anyhow::Result<tinytools::ToolResult> {
        Ok(tinytools::ToolResult::success("read result"))
    }
}

/// Native tool-to-model delivery remains absent from the durable run transcript.
#[tokio::test]
async fn native_tool_corrections_reach_the_next_request_without_entering_history() {
    use std::sync::{Arc, Mutex};
    use tinyinference_llm::{model::ModelResponse, providers::MockModel, tool::ToolCall};
    let mut response = ModelResponse::assistant("");
    response.message.content.clear();
    response.message.tool_calls = vec![ToolCall::new("read-1", "read", serde_json::json!({}))];
    response.finish_reason = Some("tool_calls".into());
    let queue = EphemeralInstructionQueue::new();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let mut harness: crate::runtime::AgentHarness<()> = crate::runtime::AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            response,
            ModelResponse::assistant("done"),
        ])),
    );
    harness.register_tool(Arc::new(ReadTool));
    harness.push_middleware(Arc::new(NativeProducer(queue.clone())));
    harness.push_middleware(Arc::new(queue.middleware()));
    harness.push_middleware(Arc::new(RequestCapture(captured.clone())));
    let run = harness
        .invoke_default(&(), vec![Message::user("read")])
        .await
        .unwrap();
    let requests = captured.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1]
            .messages
            .iter()
            .any(|message| message.text().contains("bounded correction"))
    );
    assert!(
        run.messages
            .iter()
            .all(|message| !message.text().contains("bounded correction"))
    );
    assert!(queue.drain_all().is_empty());
}

struct SkipSecondRequest(std::sync::atomic::AtomicUsize);
#[async_trait::async_trait]
impl Middleware<(), ()> for SkipSecondRequest {
    fn name(&self) -> &str {
        "skip_second_request"
    }
    async fn before_model_control(
        &self,
        _: &mut RunContext,
        _: &(),
        _: &mut ModelRequest,
    ) -> crate::error::Result<crate::context::MiddlewareControl> {
        if self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 1 {
            Ok(crate::context::MiddlewareControl::JumpTo(
                crate::context::LoopTarget::Model,
            ))
        } else {
            Ok(crate::context::MiddlewareControl::Continue)
        }
    }
}

/// A control jump abandons its request, so its correction waits for actual dispatch.
#[tokio::test]
async fn model_control_jump_keeps_corrections_for_the_next_dispatched_request() {
    use std::sync::{Arc, Mutex};
    use tinyinference_llm::{model::ModelResponse, providers::MockModel, tool::ToolCall};
    let mut response = ModelResponse::assistant("");
    response.message.content.clear();
    response.message.tool_calls = vec![ToolCall::new("read-1", "read", serde_json::json!({}))];
    response.finish_reason = Some("tool_calls".into());
    let queue = EphemeralInstructionQueue::new();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let mut harness: crate::runtime::AgentHarness<()> = crate::runtime::AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            response,
            ModelResponse::assistant("done"),
        ])),
    );
    harness.register_tool(Arc::new(ReadTool));
    harness.push_middleware(Arc::new(NativeProducer(queue.clone())));
    harness.push_middleware(Arc::new(SkipSecondRequest(
        std::sync::atomic::AtomicUsize::new(0),
    )));
    harness.push_middleware(Arc::new(queue.middleware()));
    harness.push_middleware(Arc::new(RequestCapture(captured.clone())));
    let run = harness
        .invoke_default(&(), vec![Message::user("read")])
        .await
        .unwrap();
    let requests = captured.lock().unwrap();
    assert!(
        requests
            .last()
            .unwrap()
            .messages
            .iter()
            .any(|message| message.text().contains("bounded correction"))
    );
    assert!(
        run.messages
            .iter()
            .all(|message| !message.text().contains("bounded correction"))
    );
    assert!(queue.drain_all().is_empty());
}
