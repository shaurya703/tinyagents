//! Queued instruction data and its request-scoped consumer.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// Shared producers and one-shot consumers for run-specific instructions.
///
/// Every instruction is bound to a context instance, so equal durable run ids
/// cannot mix corrections. Delivery changes only a model request copy.
#[derive(Clone, Default)]
pub struct EphemeralInstructionQueue {
    pub(super) runs: Arc<Mutex<HashMap<u64, PendingInstructions>>>,
}

/// Append the current run's queued instructions on its next model request.
///
/// Register after transcript capture and context-reduction middleware. This is
/// an observer so an earlier retry control cannot suppress the correction.
pub struct EphemeralInstructionMiddleware {
    pub(super) queue: EphemeralInstructionQueue,
}

/// A weak lifecycle marker allows cancelled/dropped contexts to be reclaimed.
#[derive(Default)]
pub(super) struct PendingInstructions {
    pub(super) lifecycle: Weak<()>,
    pub(super) instructions: Vec<String>,
}
