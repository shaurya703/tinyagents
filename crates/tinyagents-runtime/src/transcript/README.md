# Transcript execution projections

This module owns the mechanical conversion between the typed session row and
inference message contracts. It does not read files, fetch media, parse legacy
markers, choose a provider dialect, or reconcile durable history.

`transcript_to_message` keeps native assistant tool calls, tool-result correlation
ids, ordered typed media references, and nonblank reasoning verbatim. Raw tool
arguments are parsed only for execution; malformed JSON projects to null without
mutating the source row. Untyped user content remains literal text. Unknown role
strings are treated as user input.

`message_to_native_transcript` retains native tool-call structure and ordered
media parts. `message_to_display_transcript` deliberately emits a flat text history,
without assistant tool declarations or user media parts. Both omit custom host
records. Neither projection is a lossless durable codec: hosts retain raw
arguments, provider extensions, ids and metadata through `TranscriptCodec`.

`reasoning_content_block` and `reasoning_from_content` preserve nonblank reasoning
without trimming. Multiple spans remain ordered; adjacent duplicate thinking and
legacy `reasoning_content` provider-extension spans are rendered once. Assistant
rows replay this conventional reasoning field, but arbitrary metadata remains
with the host codec.

`media_from_part` and `part_from_media` only copy references. Native audio/video/
document rows preserve paths and URLs and omit prepared base64 payloads; absent
MIME defaults to `application/octet-stream`. Image rows keep their existing URL
contract, including data URLs. Preparing provider bytes is the host's operation.

```rust
use tinyagents_runtime::transcript::{transcript_to_message, message_to_native_transcript};
use tinyagents_session::transcript::TranscriptMessage;

let raw = TranscriptMessage::user("hello");
let execution = transcript_to_message(&raw);
let projected = message_to_native_transcript(&execution).unwrap();
assert_eq!(projected.content, "hello");
```

Nine tests cover native correlation, malformed arguments, ordered media,
ephemeral provider bytes, reasoning replay, literal legacy markers, empty parts,
custom-record omission and unknown inbound roles. Lossless durable dialect and
legacy marker behavior remain host integration responsibilities.

`native_transcript_to_messages` checks the complete native sequence before
replay. Orphan/duplicate results, missing results and interrupted tool rounds
return a codec error identifying the call or row. Display rows preserve product
result identities but omit declarations, so they must never be used as native
replay input. `transcript_to_message` is only a low-level single-row projection.
