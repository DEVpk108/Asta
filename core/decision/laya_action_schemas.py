ACTION_TYPES = {
    "open_app": "Open, launch, switch to, or bring an installed application to the foreground.",
    "close_app": "Close or quit an installed application that is currently running.",
    "open_url": "Open a website or web URL without performing a search.",
    "web_search": "Search the web for something, optionally using a named search site.",
    "open_folder": "Open a local folder such as Downloads, Desktop, Documents, or another directory.",
    "type_text": "Type, write, enter, or paste text into the currently focused application.",
    "keypress": "Press a keyboard key or shortcut such as enter, escape, copy, paste, save, or new tab.",
    "scroll": "Scroll the current page or document up, down, to the top, or to the bottom.",
    "screenshot": "Capture a screenshot of the current screen.",
    "media": "Control media playback such as play, pause, next track, or previous track.",
    "volume": "Change system audio volume, mute, or unmute.",
    "system": "Perform a supported operating-system action.",
    "task": "Perform a multi-step computer workflow requiring several actions.",
    "none": "Conversation, a question, background speech, or text that is not a computer command.",
}


def build_action_questions(applications: list[str]) -> dict:
    apps = {
        name: f"Installed application named {name}."
        for name in applications
        if str(name).strip()
    }
    apps["none"] = "No installed application matches the user's target."

    return {
        "action": {
            "type": "choice",
            "instructions": (
                "Which single computer action is the user currently requesting? "
                "Choose from the finite action vocabulary. Do not invent a new action."
            ),
            "criteria": ACTION_TYPES,
        },
        "addressed": {
            "type": "noul",
            "instructions": (
                "Is the utterance directly addressed to A.S.T.A. as a computer-control "
                "instruction rather than casual conversation, a question, or background speech?"
            ),
            "criteria": {
                "true": "A direct instruction for A.S.T.A. to control the computer.",
                "false": "Not a direct computer-control instruction.",
            },
        },
        "compound": {
            "type": "noul",
            "instructions": (
                "Does the utterance request two or more distinct computer actions "
                "that should occur sequentially?"
            ),
            "criteria": {
                "true": "Two or more separate actions are requested.",
                "false": "Exactly one action is requested.",
            },
        },
        "command_complete": {
            "type": "noul",
            "instructions": (
                "Is the current utterance complete enough to execute the requested "
                "action now, or does it clearly continue into more speech?"
            ),
            "criteria": {
                "true": "The current command is complete and actionable now.",
                "false": "The speaker is still building the command or its target.",
            },
        },
        "target_app": {
            "type": "choice",
            "instructions": (
                "Which installed application does the user refer to? "
                "Match natural aliases and abbreviations to the closest installed application. "
                "Choose none when no listed application matches."
            ),
            "criteria": apps,
        },
    }
