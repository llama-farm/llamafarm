from routers.chat_completions.service import ChatCompletionsService


def test_normalize_logprobs_openai_shape_passthrough():
    payload = {
        "content": [
            {
                "token": "goto_waypoint",
                "logprob": -0.06,
                "bytes": [103, 111],
                "top_logprobs": [
                    {"token": "goto_waypoint", "logprob": -0.06, "bytes": [103]},
                    {"token": "move_relative", "logprob": -0.68, "bytes": [109]},
                ],
            }
        ]
    }

    normalized = ChatCompletionsService._normalize_logprobs_payload(payload, 2)
    assert normalized == payload


def test_normalize_logprobs_legacy_shape_conversion():
    payload = {
        "tokens": ["goto_waypoint", "{"],
        "token_logprobs": [-0.06, -0.20],
        "top_logprobs": [
            {"goto_waypoint": -0.06, "move_relative": -0.68},
            {"{": -0.20, "[": -1.1},
        ],
    }

    normalized = ChatCompletionsService._normalize_logprobs_payload(payload, 1)

    assert normalized is not None
    content = normalized["content"]
    assert len(content) == 2
    assert content[0]["token"] == "goto_waypoint"
    assert content[0]["logprob"] == -0.06
    assert content[0]["top_logprobs"][0]["token"] == "goto_waypoint"
    assert len(content[0]["top_logprobs"]) == 1
