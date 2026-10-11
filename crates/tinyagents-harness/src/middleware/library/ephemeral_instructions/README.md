# Ephemeral instructions

`EphemeralInstructionQueue` connects a host correction producer to the next
model request. Bind each instruction to the producing `RunContext`; a shared
queue never delivers it to another invocation, even when durable run ids match.

Install `queue.middleware()` before dispatch observers, and call
`queue.push(context, instruction)` from a tool or other lifecycle hook. Delivery
uses the provider profile through `push_ephemeral_instruction`, preserving
hoisting models' system prefix. Instructions modify the outgoing request copy
and do not enter the returned run transcript. Delivery consumes each instruction
once, in insertion order.

Success and error hooks discard unsent instructions for that context. Admission
reclaims contexts dropped without a terminal hook. `drain_for` is the explicit
per-context consumer; `drain_all` is administrative shutdown/inspection and has
no ordering guarantee between contexts. Clones share the queue.

The queue does not classify failures, choose corrective text, or decide whether
to retry or halt. Those policies remain with its producer. It performs no I/O.

Delivery runs at `before_model_dispatch`, after admission/control checkpoints
and final model resolution. A skipped request does not consume instructions.
Ordinary `before_model` hooks and transcript snapshots run before this boundary;
observers requiring the final ephemeral request use the dispatch hook instead.
