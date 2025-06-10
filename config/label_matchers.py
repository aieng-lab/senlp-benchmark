"""Label matchers.

Here is detailed how the class labels need to be matched from numbers to texts for all the
evaluation datasets.
"""

matchers = {
    "bug_issue": {
        0: "not a bug",
        1: "bug"
    },
    "closed_question": {
        0: "open",
        1: "not a real question",
        2: "off topic",
        3: "not constructive",
        4: "too localized"
    },
    "comment_type_java": [
        "summary",
        "ownership",
        "expand",
        "usage",
        "pointer",
        "deprecation",
        "rational"
    ],
    "comment_type_python": [
        "usage",
        "parameters",
        "developmentNotes",
        "expand",
        "Summary"
    ],
    "comment_type_pharo": [
        "keyimplementationpoints",
        "example",
        "responsibilities",
        "classreferences",
        "intent",
        "keymessages",
        "collaborators"
    ],
    "commit_intent": {
        0: "other",
        1: "perfective",
        2: "corrective"
    },
    "incivility": {
        0: "uncivil",
        1: "civil"
    },
    "issue_type": {
        0: "bug",
        1: "enhancement",
        2: "question"
    },
    "question_quality": {
        0: "LQ_CLOSE",
        1: "LQ_EDIT",
        2: "HQ"
    },
    "requirement_completion": [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SPACE",
        "SYM",
        "VERB",
        "X"
    ],
    "requirement_type": {
        0: "non-functional",
        1: "functional"
    },
    "review_aspect": [
        "usability",
        "others",
        "onlysentiment",
        "bug",
        "performance",
        "community",
        "documentation",
        "compatibility",
        "legal",
        "portability",
        "security"
    ],
    "se_entities": [
        "O",
        "B-Data_Structure", "I-Data_Structure",
        "B-Application", "I-Application",
        "B-Code_Block", "I-Code_Block",
        "B-Function", "I-Function",
        "B-Data_Type", "I-Data_Type",
        "B-Language", "I-Language",
        "B-Library", "I-Library",
        "B-Variable", "I-Variable",
        "B-Device", "I-Device",
        "B-User_Name", "I-User_Name",
        "B-User_Interface_Element", "I-User_Interface_Element",
        "B-Class", "I-Class",
        "B-Website", "I-Website",
        "B-Version", "I-Version",
        "B-File_Name", "I-File_Name",
        "B-File_Type", "I-File_Type",
        "B-Operating_System", "I-Operating_System",
        "B-Output_Block", "I-Output_Block",
        "B-Algorithm", "I-Algorithm",
        "B-HTML_XML_Tag", "I-HTML_XML_Tag"
    ],
    "sentiment": {
        0: "negative",
        1: "neutral",
        2: "positive"
    },
    "smell_doc": [
        "fragmented",
        "tangled",
        "excessive",
        "bloated",
        "lazy"
    ],
    "story_points": None,
    "tone_bearing": {
        0: "technical",
        1: "non-technical",
    }
}
