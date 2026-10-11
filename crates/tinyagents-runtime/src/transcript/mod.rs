//! Mechanical execution projections over durable transcript rows.
//!
//! These helpers do not replace a lossless host [`crate::TranscriptCodec`].
//! They intentionally project provider-neutral execution fields; hosts retain
//! durable dialects, raw argument strings, metadata, and legacy marker policy.
use tinyagents_session::transcript::{
    TranscriptMediaRef, TranscriptMessage, TranscriptPart, TranscriptToolCall,
};
use tinyinference_llm::message::{
    AssistantMessage, ContentBlock, ImageRef, MediaRef, Message, SystemMessage, ToolMessage,
    UserMessage,
};
use tinyinference_llm::tool::ToolCall as TaToolCall;

/// Conventional provider reasoning field carried by durable assistant rows.
pub const REASONING_EXT_KEY: &str = "reasoning_content";

/// Preserve a nonblank reasoning span verbatim for execution replay.
pub fn reasoning_content_block(reasoning: Option<&str>) -> Option<ContentBlock> {
    let reasoning = reasoning?;
    // Store verbatim (only gate on non-empty after a trim): thinking-mode
    // providers validate the prior reasoning block byte-for-byte on a resumed
    // multi-turn request, so trimming boundary whitespace could break replay.
    (!reasoning.trim().is_empty()).then(|| ContentBlock::Thinking {
        text: reasoning.to_string(),
        signature: None,
    })
}

/// Separator between distinct thinking blocks joined by
/// [`reasoning_from_content`].
const REASONING_BLOCK_SEPARATOR: &str = "\n\n";

/// Recover `reasoning_content` from an assistant message's content blocks.
///
/// A message can carry several thinking blocks (interleaved thinking emits one
/// per reasoning span). All of them are kept, in order, joined by a blank
/// line — keeping only the first silently dropped every later span from the
/// persisted transcript and the reasoning shown for that step.
pub fn reasoning_from_content(content: &[ContentBlock]) -> Option<String> {
    let mut parts: Vec<String> = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Thinking { text, .. } => Some(text.clone()),
            ContentBlock::ProviderExtension(value) => value
                .get(REASONING_EXT_KEY)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
        .collect();
    // A legacy row can carry the same reasoning both as a thinking block and
    // under the provider-extension key; do not render it twice.
    parts.dedup();
    match parts.len() {
        0 => None,
        1 => parts.into_iter().next(),
        _ => Some(parts.join(REASONING_BLOCK_SEPARATOR)),
    }
}

/// The `extra_metadata` an assistant [`TranscriptMessage`] should carry so
/// `reasoning_content` replays on the next provider request.
fn reasoning_extra_metadata(content: &[ContentBlock]) -> Option<serde_json::Value> {
    reasoning_from_content(content)
        .map(|reasoning| serde_json::json!({ REASONING_EXT_KEY: reasoning }))
}

/// Convert one transcript row into a harness [`Message`].
///
/// Role strings map onto the typed arms: an assistant row's `tool_calls` become
/// [`AssistantMessage::tool_calls`], a tool row's `tool_call_id` becomes
/// [`ToolMessage::tool_call_id`] (falling back to the row id, then an empty id
/// for a bare tool message), and a user row's `parts` become text and
/// [`ContentBlock::Image`] blocks in order. Keeping the calls on the assistant
/// message is what stops the harness re-sending orphan `tool` messages, which
/// native providers reject (`assistant message with 'tool_calls' must be
/// followed by tool messages`).
/// Convert one typed durable row to a provider-neutral execution message.
/// Untyped user text stays literal; hosts apply any legacy marker policy.
pub fn transcript_to_message(msg: &TranscriptMessage) -> Message {
    let text = msg.content.clone();
    match msg.role.as_str() {
        "system" => Message::System(SystemMessage {
            content: vec![ContentBlock::Text(text)],
            sections: Default::default(),
            tools_added: Vec::new(),
            tools_removed: Vec::new(),
        }),
        "assistant" => {
            // Restore any `reasoning_content` stashed on the persisted message so a
            // multi-turn thinking-mode conversation replays it verbatim (see
            // [`reasoning_content_block`]).
            let reasoning = msg
                .extra_metadata
                .as_ref()
                .and_then(|meta| meta.get(REASONING_EXT_KEY))
                .and_then(serde_json::Value::as_str);
            let mut content = vec![ContentBlock::Text(text)];
            content.extend(reasoning_content_block(reasoning));
            Message::Assistant(AssistantMessage {
                id: msg.id.clone(),
                content,
                tool_calls: msg.tool_calls.iter().map(row_call_to_ta_call).collect(),
                usage: None,
                origin: None,
            })
        }
        "tool" => Message::Tool(ToolMessage {
            tool_call_id: msg
                .tool_call_id
                .clone()
                .or_else(|| msg.id.clone())
                .unwrap_or_default(),
            content: vec![ContentBlock::Text(text)],
            trusted_verbatim: false,
            artifact: None,
        }),
        // "user" and any unrecognized role default to a user turn — the safest
        // mapping for a free-form inbound message.
        _ => Message::User(UserMessage {
            content: match msg.parts.as_deref() {
                Some(parts) => user_blocks_from_parts(parts),
                None => vec![ContentBlock::Text(text)],
            },
        }),
    }
}

/// The content blocks of a user row's typed `parts`, in source order.
fn user_blocks_from_parts(parts: &[TranscriptPart]) -> Vec<ContentBlock> {
    let blocks: Vec<ContentBlock> = parts
        .iter()
        .map(|part| match part {
            TranscriptPart::Text { text } => ContentBlock::Text(text.clone()),
            TranscriptPart::Image { url } => ContentBlock::Image(ImageRef {
                url: url.clone(),
                mime_type: data_uri_mime(url),
            }),
            TranscriptPart::Audio { source, mime_type } => {
                ContentBlock::Audio(media_from_part(source, mime_type))
            }
            TranscriptPart::Video { source, mime_type } => {
                ContentBlock::Video(media_from_part(source, mime_type))
            }
            TranscriptPart::Document { source, mime_type } => {
                ContentBlock::Document(media_from_part(source, mime_type))
            }
        })
        .collect();
    if blocks.is_empty() {
        vec![ContentBlock::Text(String::new())]
    } else {
        blocks
    }
}

fn data_uri_mime(reference: &str) -> Option<String> {
    let rest = reference.strip_prefix("data:")?;
    let mime = rest.split([';', ',']).next()?.trim();
    (!mime.is_empty()).then(|| mime.to_string())
}

/// Rebuild a harness [`TaToolCall`] from a row's [`TranscriptToolCall`] (whose
/// `arguments` is the serialized JSON string the provider emitted).
fn row_call_to_ta_call(call: &TranscriptToolCall) -> TaToolCall {
    TaToolCall {
        id: call.id.clone(),
        name: call.name.clone(),
        arguments: serde_json::from_str(&call.arguments).unwrap_or(serde_json::Value::Null),
        invalid: None,
    }
}

/// A harness [`TaToolCall`] as a row's [`TranscriptToolCall`].
fn ta_call_to_row_call(call: &TaToolCall) -> TranscriptToolCall {
    TranscriptToolCall {
        id: call.id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.to_string(),
        extra_content: None,
    }
}

pub fn message_to_flat_transcript(msg: &Message) -> Option<TranscriptMessage> {
    Some(match msg {
        Message::System(_) => TranscriptMessage::system(msg.text()),
        Message::User(_) => TranscriptMessage::user(msg.text()),
        Message::Assistant(a) => {
            let mut cm = TranscriptMessage::assistant(msg.text());
            cm.extra_metadata = reasoning_extra_metadata(&a.content);
            cm
        }
        Message::Tool(t) => {
            let mut cm = TranscriptMessage::tool(msg.text());
            cm.id = Some(t.tool_call_id.clone());
            cm
        }
        Message::Custom(c) => {
            tracing::trace!(kind = %c.kind, "dropping custom execution projection");
            return None;
        }
    })
}

/// Convert one harness [`Message`] into a typed row for a **native**
/// tool-calling provider request, preserving the structure the provider needs
/// to round-trip a tool round: an assistant turn that made tool calls keeps
/// them in `tool_calls`, a tool result keeps its `tool_call_id`, and a user turn
/// with pictures keeps its text and image blocks as ordered `parts`. Without
/// this the provider sees an assistant with no `tool_calls` followed by an
/// orphan tool message and drops the round — breaking multi-turn native tool
/// calling (e.g. the orchestrator's `spawn_parallel_agents` → synthesis hop) —
/// and a native-tool provider silently loses every pasted image.
///
/// Returns `None` for [`Message::Custom`]: that variant is a host-side
/// out-of-band record (compaction marker, label, audit note) that the harness
/// never sends to a provider, so it must not leak into the next request.
pub fn message_to_native_transcript(msg: &Message) -> Option<TranscriptMessage> {
    Some(match msg {
        Message::System(_) => TranscriptMessage::system(msg.text()),
        Message::User(user) => user_row(msg, user),
        Message::Assistant(a) => {
            let mut cm = TranscriptMessage::assistant_with_calls(
                msg.text(),
                a.tool_calls.iter().map(ta_call_to_row_call).collect(),
            );
            cm.extra_metadata = reasoning_extra_metadata(&a.content);
            cm
        }
        Message::Tool(t) => {
            let mut cm = TranscriptMessage::tool_result(t.tool_call_id.clone(), msg.text());
            cm.id = Some(t.tool_call_id.clone());
            cm
        }
        Message::Custom(c) => {
            tracing::trace!(kind = %c.kind, "dropping custom execution projection");
            return None;
        }
    })
}

/// The row of a user message: its text, plus ordered parts when it carries an
/// image. Json / provider-extension blocks carry no user-visible text, so they
/// are dropped, as [`Message::text`] drops them.
fn user_row(msg: &Message, user: &UserMessage) -> TranscriptMessage {
    let has_image = user.content.iter().any(|block| {
        matches!(
            block,
            ContentBlock::Image(_)
                | ContentBlock::Audio(_)
                | ContentBlock::Video(_)
                | ContentBlock::Document(_)
        )
    });
    if !has_image {
        return TranscriptMessage::user(msg.text());
    }
    TranscriptMessage::user_with_parts(
        user.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) if !text.is_empty() => {
                    Some(TranscriptPart::Text { text: text.clone() })
                }
                ContentBlock::Image(image) => Some(TranscriptPart::Image {
                    url: image.url.clone(),
                }),
                ContentBlock::Audio(media) => part_from_media(media, "audio"),
                ContentBlock::Video(media) => part_from_media(media, "video"),
                ContentBlock::Document(media) => part_from_media(media, "document"),
                _ => None,
            })
            .collect(),
    )
}

/// Project a durable path or URL into an unresolved inference media reference.
pub fn media_from_part(source: &TranscriptMediaRef, mime: &str) -> MediaRef {
    match source {
        TranscriptMediaRef::Path { path } => MediaRef::Path {
            path: path.clone(),
            media_type: Some(mime.into()),
        },
        TranscriptMediaRef::Url { url } => MediaRef::Url {
            url: url.clone(),
            media_type: Some(mime.into()),
        },
    }
}

/// Project unresolved media into a durable part; inline provider bytes return None.
/// Kinds audio/video select those variants; other kinds select a document.
pub fn part_from_media(media: &MediaRef, kind: &str) -> Option<TranscriptPart> {
    let (source, mime_type) = match media {
        MediaRef::Path { path, media_type } => (
            TranscriptMediaRef::Path { path: path.clone() },
            media_type.clone(),
        ),
        MediaRef::Url { url, media_type } => (
            TranscriptMediaRef::Url { url: url.clone() },
            media_type.clone(),
        ),
        // Inline payloads are provider-only and must never become durable rows.
        MediaRef::Base64 { .. } => return None,
    };
    let mime_type = mime_type.unwrap_or_else(|| "application/octet-stream".into());
    Some(match kind {
        "audio" => TranscriptPart::Audio { source, mime_type },
        "video" => TranscriptPart::Video { source, mime_type },
        _ => TranscriptPart::Document { source, mime_type },
    })
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
