use super::*;
use tinyagents_session::transcript::{
    TranscriptMediaRef, TranscriptMessage, TranscriptPart, TranscriptToolCall,
};
use tinyinference_llm::message::{ContentBlock, MediaRef, Message, UserMessage};

/// Native projection retains the assistant/tool pair and correlation fallback.
#[test]
fn native_tool_round_preserves_calls_and_tool_result_identity() {
    let row = TranscriptMessage::assistant_with_calls(
        "running",
        vec![TranscriptToolCall {
            id: "call-1".into(),
            name: "read".into(),
            arguments: "{\"path\":\"a\"}".into(),
            extra_content: None,
        }],
    );
    let message = transcript_to_message(&row);
    let native = message_to_native_transcript(&message).unwrap();
    assert_eq!(native.tool_calls, row.tool_calls);
    assert!(
        message_to_display_transcript(&message)
            .unwrap()
            .tool_calls
            .is_empty()
    );
    let mut tool = TranscriptMessage::tool("result");
    tool.id = Some("call-1".into());
    let message = transcript_to_message(&tool);
    let native = message_to_native_transcript(&message).unwrap();
    assert_eq!(native.tool_call_id.as_deref(), Some("call-1"));
    assert_eq!(native.id.as_deref(), Some("call-1"));
}

/// Execution parsing must not mutate the raw durable arguments it reads.
#[test]
fn malformed_arguments_project_to_null_without_changing_the_source_row() {
    let row = TranscriptMessage::assistant_with_calls(
        "",
        vec![TranscriptToolCall {
            id: "call".into(),
            name: "read".into(),
            arguments: "{unfinished".into(),
            extra_content: Some(serde_json::json!({"provider": "raw"})),
        }],
    );
    let Message::Assistant(message) = transcript_to_message(&row) else {
        panic!("assistant");
    };
    assert_eq!(message.tool_calls[0].arguments, serde_json::Value::Null);
    assert_eq!(row.tool_calls[0].arguments, "{unfinished");
    assert_eq!(
        row.tool_calls[0].extra_content,
        Some(serde_json::json!({"provider": "raw"}))
    );
}

/// Typed media retain their original order and references without reading files.
#[test]
fn ordered_user_parts_round_trip_as_references() {
    let row = TranscriptMessage::user_with_parts(vec![
        TranscriptPart::Text {
            text: "first".into(),
        },
        TranscriptPart::Image {
            url: "data:image/png;base64,AA==".into(),
        },
        TranscriptPart::Audio {
            source: TranscriptMediaRef::Path {
                path: "clip.wav".into(),
            },
            mime_type: "audio/wav".into(),
        },
        TranscriptPart::Video {
            source: TranscriptMediaRef::Url {
                url: "https://example.invalid/movie".into(),
            },
            mime_type: "video/mp4".into(),
        },
        TranscriptPart::Document {
            source: TranscriptMediaRef::Path {
                path: "guide.pdf".into(),
            },
            mime_type: "application/pdf".into(),
        },
        TranscriptPart::Text {
            text: "last".into(),
        },
    ]);
    let message = transcript_to_message(&row);
    let native = message_to_native_transcript(&message).unwrap();
    assert_eq!(native.parts, row.parts);
    let Message::User(user) = message else {
        panic!("user");
    };
    let ContentBlock::Image(image) = &user.content[1] else {
        panic!("image");
    };
    assert_eq!(image.mime_type.as_deref(), Some("image/png"));
}

/// Provider-prepared bytes are ephemeral; only paths and URLs enter durable parts.
#[test]
fn native_media_projection_omits_inline_provider_bytes() {
    let message = Message::User(UserMessage {
        content: vec![
            ContentBlock::Text("read".into()),
            ContentBlock::Document(MediaRef::Base64 {
                data: "private-inline-data".into(),
                media_type: "application/pdf".into(),
            }),
            ContentBlock::Audio(MediaRef::Path {
                path: "sound".into(),
                media_type: None,
            }),
        ],
    });
    let native = message_to_native_transcript(&message).unwrap();
    assert_eq!(
        native.parts.unwrap(),
        vec![
            TranscriptPart::Text {
                text: "read".into()
            },
            TranscriptPart::Audio {
                source: TranscriptMediaRef::Path {
                    path: "sound".into()
                },
                mime_type: "application/octet-stream".into()
            },
        ]
    );
}

/// Thinking replay preserves byte boundaries and all distinct reasoning spans.
#[test]
fn reasoning_replays_verbatim_and_retains_multiple_spans() {
    let mut row = TranscriptMessage::assistant("answer");
    row.extra_metadata = Some(serde_json::json!({"reasoning_content": "  thought\n"}));
    let message = transcript_to_message(&row);
    let native = message_to_native_transcript(&message).unwrap();
    assert_eq!(native.extra_metadata, row.extra_metadata);
    let content = vec![
        ContentBlock::Thinking {
            text: " first ".into(),
            signature: None,
        },
        ContentBlock::ProviderExtension(serde_json::json!({"reasoning_content": " first "})),
        ContentBlock::Thinking {
            text: "second".into(),
            signature: None,
        },
    ];
    assert_eq!(
        reasoning_from_content(&content).as_deref(),
        Some(" first \n\nsecond")
    );
    assert!(reasoning_content_block(Some(" \n ")).is_none());
}

/// Marker lifting is a host compatibility policy, not this typed conversion core.
#[test]
fn untyped_user_text_is_preserved_without_interpreting_legacy_markers() {
    let row = TranscriptMessage::user("literal [IMAGE:old-path] and [FILE:file]");
    assert_eq!(transcript_to_message(&row).text(), row.content);
    let row = TranscriptMessage::user_with_parts(Vec::new());
    let Message::User(user) = transcript_to_message(&row) else {
        panic!("user");
    };
    assert_eq!(user.content.len(), 1);
    assert!(matches!(&user.content[0], ContentBlock::Text(text) if text.is_empty()));
}

/// Out-of-band host records must never leak into a later provider request.
#[test]
fn custom_records_are_omitted_from_both_execution_projections() {
    let custom = Message::Custom(tinyinference_llm::message::CustomMessage {
        kind: "audit".into(),
        payload: serde_json::json!({"private": true}),
        display: Some("note".into()),
    });
    assert!(message_to_display_transcript(&custom).is_none());
    assert!(message_to_native_transcript(&custom).is_none());
    let mut row = TranscriptMessage::user("inbound");
    row.role = "unknown".into();
    assert!(matches!(transcript_to_message(&row), Message::User(_)));
}

/// Native replay keeps a complete declaration/result pair correlated after projection.
#[test]
fn complete_native_tool_round_remains_valid_after_projection_and_replay() {
    let Message::Assistant(mut assistant) = Message::assistant("") else {
        unreachable!()
    };
    assistant.tool_calls.push(TaToolCall::new(
        "call-1",
        "read",
        serde_json::json!({"path":"a"}),
    ));
    let history = vec![
        Message::Assistant(assistant),
        Message::tool("call-1", "result"),
    ];
    let rows: Vec<_> = history
        .iter()
        .filter_map(message_to_native_transcript)
        .collect();
    let replay: Vec<_> = rows.iter().map(transcript_to_message).collect();
    let Message::Assistant(assistant) = &replay[0] else {
        panic!("assistant declaration");
    };
    let Message::Tool(tool) = &replay[1] else {
        panic!("tool result");
    };
    assert_eq!(assistant.tool_calls.len(), 1);
    assert_eq!(assistant.tool_calls[0].id, tool.tool_call_id);
    assert_eq!(
        assistant.tool_calls[0].arguments,
        serde_json::json!({"path":"a"})
    );
    assert_eq!(replay, history);
}

#[test]
fn native_history_decode_rejects_display_rows_and_unanswered_or_duplicate_results() {
    let Message::Assistant(mut assistant) = Message::assistant("") else {
        unreachable!()
    };
    assistant.tool_calls = vec![TaToolCall::new("call-1", "read", serde_json::json!({}))];
    let messages = vec![
        Message::Assistant(assistant),
        Message::tool("call-1", "result"),
    ];
    let native: Vec<_> = messages
        .iter()
        .filter_map(message_to_native_transcript)
        .collect();
    assert_eq!(native_transcript_to_messages(&native).unwrap(), messages);
    let display: Vec<_> = messages
        .iter()
        .filter_map(message_to_display_transcript)
        .collect();
    assert!(
        native_transcript_to_messages(&display)
            .unwrap_err()
            .to_string()
            .contains("orphan")
    );
    assert!(
        native_transcript_to_messages(&native[..1])
            .unwrap_err()
            .to_string()
            .contains("unanswered")
    );
    let mut duplicate = native.clone();
    duplicate.push(native[1].clone());
    assert!(
        native_transcript_to_messages(&duplicate)
            .unwrap_err()
            .to_string()
            .contains("orphan")
    );
    let mut interrupted = native.clone();
    interrupted.insert(1, TranscriptMessage::user("interrupt"));
    assert!(native_transcript_to_messages(&interrupted).is_err());
}
