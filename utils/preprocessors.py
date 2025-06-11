"""ETL pre-processing pipelines per dataset.

This module specifies the pipelines to be applied to the different datasets
and their respective entities. The pipelines need to be defined using the
Scikit-Learn Pipeline interface.
"""

from sklearn.pipeline import Pipeline

from .preprocessing_steps import (
    BasicPreprocessor,
    HTMLExtractor,
    Jira2MarkdownConverter,
    RegexMasker,
)


pretraining_pipelines = {
    "cased": {
        "arxiv": {
            "papers": {
                "title": Pipeline([
                    ("basic", BasicPreprocessor(lower=False, remove_newlines=True))
                ]),
                "abstract": Pipeline([
                    ("basic", BasicPreprocessor(lower=False, remove_newlines=True)),
                    ("url", RegexMasker(pattern="URL"))
                ])
            }
        },
        "github": {
            "issues": {
                "title": Pipeline([
                    ("basic", BasicPreprocessor(lower=False))
                ]),
                "body": Pipeline([
                    ("basic", BasicPreprocessor(lower=False)),
                    ("html", HTMLExtractor(in_markdown=True)),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            },
            "commits": {
                "messages": Pipeline([
                    ("basic", BasicPreprocessor(lower=False)),
                    ("html", HTMLExtractor(in_markdown=True)),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            },
            "comments": {
                "body": Pipeline([
                    ("basic", BasicPreprocessor(lower=False)),
                    ("html", HTMLExtractor(in_markdown=True)),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            }
        },
        "jira": {
            "issues": {
                "summary": Pipeline([
                    ("basic", BasicPreprocessor(lower=False))
                ]),
                "description": Pipeline([
                    ("basic", BasicPreprocessor(lower=False)),
                    ("jira", Jira2MarkdownConverter()),
                    ("html", HTMLExtractor(in_markdown=True)),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            },
            "comments": {
                "body": Pipeline([
                    ("basic", BasicPreprocessor(lower=False)),
                    ("jira", Jira2MarkdownConverter()),
                    ("html", HTMLExtractor(in_markdown=True)),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            }
        },
        "stackoverflow": {
            "posts": {
                "title": Pipeline([
                    ("basic", BasicPreprocessor(lower=False))
                ]),
                "body": Pipeline([
                    ("basic", BasicPreprocessor(lower=False)),
                    ("html", HTMLExtractor()),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            },
            "comments": {
                "text": Pipeline([
                    ("basic", BasicPreprocessor(lower=False)),
                    ("html", HTMLExtractor()),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            }
        }
    },
    "uncased": {
        "arxiv": {
            "papers": {
                "title": Pipeline([
                    ("basic", BasicPreprocessor(remove_newlines=True))
                ]),
                "abstract": Pipeline([
                    ("basic", BasicPreprocessor(remove_newlines=True)),
                    ("url", RegexMasker(pattern="URL"))
                ])
            }
        },
        "github": {
            "issues": {
                "title": Pipeline([
                    ("basic", BasicPreprocessor())
                ]),
                "body": Pipeline([
                    ("basic", BasicPreprocessor()),
                    ("html", HTMLExtractor(in_markdown=True)),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            },
            "commits": {
                "messages": Pipeline([
                    ("basic", BasicPreprocessor()),
                    ("html", HTMLExtractor(in_markdown=True)),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            },
            "comments": {
                "body": Pipeline([
                    ("basic", BasicPreprocessor()),
                    ("html", HTMLExtractor(in_markdown=True)),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            }
        },
        "jira": {
            "issues": {
                "summary": Pipeline([
                    ("basic", BasicPreprocessor())
                ]),
                "description": Pipeline([
                    ("basic", BasicPreprocessor()),
                    ("jira", Jira2MarkdownConverter()),
                    ("html", HTMLExtractor(in_markdown=True)),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            },
            "comments": {
                "body": Pipeline([
                    ("basic", BasicPreprocessor()),
                    ("jira", Jira2MarkdownConverter()),
                    ("html", HTMLExtractor(in_markdown=True)),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            }
        },
        "stackoverflow": {
            "posts": {
                "title": Pipeline([
                    ("basic", BasicPreprocessor())
                ]),
                "body": Pipeline([
                    ("basic", BasicPreprocessor()),
                    ("html", HTMLExtractor()),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            },
            "comments": {
                "text": Pipeline([
                    ("basic", BasicPreprocessor()),
                    ("html", HTMLExtractor()),
                    ("code", RegexMasker(pattern="CODE")),
                    ("hash", RegexMasker(pattern="HASH")),
                    ("url", RegexMasker(pattern="URL")),
                    ("user", RegexMasker(pattern="USER"))
                ])
            }
        }
    }
}

evaluation_pipelines = {
    "cased": {
        "bug_issue": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("jira", Jira2MarkdownConverter()),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "closed_question": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "comment_type": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("url", RegexMasker(pattern="URL"))
        ]),
        "commit_intent": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "incivility": Pipeline([
            ("basic", BasicPreprocessor(lower=False))
        ]),
        "issue_type": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "post_api": Pipeline([
            ("basic", BasicPreprocessor(lower=False))
        ]),
        "post_tag": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("html", HTMLExtractor(in_markdown=False)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER")),
            ("non-utf8", RegexMasker(pattern="NON-UTF8"))
        ]),
        "question_quality": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("html", HTMLExtractor(in_markdown=False)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "requirement_type": Pipeline([
            ("basic", BasicPreprocessor(lower=False))
        ]),
        "review_aspect": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "sentiment": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "smell_doc": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("url", RegexMasker(pattern="URL")),
        ]),
        "story_points": Pipeline([
            ("basic", BasicPreprocessor(lower=False)),
            ("jira", Jira2MarkdownConverter()),
            ("html", HTMLExtractor(in_markdown=True)),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "tone_bearing": Pipeline([
            ("basic", BasicPreprocessor(lower=False))
        ])
    },
    "uncased": {
        "bug_issue": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("jira", Jira2MarkdownConverter()),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "closed_question": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "comment_type": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("url", RegexMasker(pattern="URL"))
        ]),
        "commit_intent": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "incivility": Pipeline([
            ("basic", BasicPreprocessor(lower=True))
        ]),
        "issue_type": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "post_api": Pipeline([
            ("basic", BasicPreprocessor(lower=True))
        ]),
        "post_tag": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("html", HTMLExtractor(in_markdown=False)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER")),
            ("non-utf8", RegexMasker(pattern="NON-UTF8"))
        ]),
        "question_quality": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("html", HTMLExtractor(in_markdown=False)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "requirement_type": Pipeline([
            ("basic", BasicPreprocessor(lower=True))
        ]),
        "review_aspect": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "sentiment": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("hash", RegexMasker(pattern="HASH")),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "smell_doc": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("html", HTMLExtractor(in_markdown=True)),
            ("code", RegexMasker(pattern="CODE")),
            ("url", RegexMasker(pattern="URL")),
        ]),
        "story_points": Pipeline([
            ("basic", BasicPreprocessor(lower=True)),
            ("jira", Jira2MarkdownConverter()),
            ("html", HTMLExtractor(in_markdown=True)),
            ("url", RegexMasker(pattern="URL")),
            ("user", RegexMasker(pattern="USER"))
        ]),
        "tone_bearing": Pipeline([
            ("basic", BasicPreprocessor(lower=True))
        ])
    }
}
