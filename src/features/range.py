"""
Feature-Based Range (Vocabulary) Assessment

Extracts interpretable vocabulary features from transcription text and
converts them to CEFR levels based on established research benchmarks.

Range Sub-dimensions:
1. Lexical Diversity - Type-Token Ratio, unique words
2. Word Sophistication - Word frequency, academic words
3. Vocabulary Breadth - Word length, syllable complexity
4. Lexical Density - Content vs function words

References:
- CEFR Companion Volume (2020)
- Laufer & Nation (1995) - Lexical richness measures
- Coxhead (2000) - Academic Word List
- Read (2000) - Assessing Vocabulary

Example:
    >>> from src.features.range import RangeAssessor
    >>> assessor = RangeAssessor()
    >>> result = assessor.assess("I think learning English is very important...")
    >>> print(result.level)  # "B1"
    >>> print(result.sub_scores)  # {"diversity": 72.5, "sophistication": 65.0, ...}
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter

# Import pure calculation functions from calculations.py
from .calculations import (
    calculate_ttr,
    calculate_root_ttr,
    calculate_corrected_ttr,
    calculate_hapax_ratio,
    calculate_lexical_density,
    calculate_avg_word_length,
    calculate_long_word_ratio,
    count_syllables,
    tokenize,
    get_word_frequencies,
    score_to_cefr_level,
)


# Common English words (high frequency - A1/A2 level)
# Based on General Service List (GSL) top 500
COMMON_WORDS = {
    # Articles, pronouns, prepositions
    "a",
    "an",
    "the",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "who",
    "what",
    "which",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "of",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    # Common verbs
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "go",
    "goes",
    "went",
    "gone",
    "going",
    "come",
    "comes",
    "came",
    "get",
    "gets",
    "got",
    "getting",
    "make",
    "makes",
    "made",
    "making",
    "know",
    "knows",
    "knew",
    "known",
    "think",
    "thinks",
    "thought",
    "see",
    "sees",
    "saw",
    "seen",
    "want",
    "wants",
    "wanted",
    "use",
    "uses",
    "used",
    "find",
    "finds",
    "found",
    "give",
    "gives",
    "gave",
    "tell",
    "tells",
    "told",
    "say",
    "says",
    "said",
    "take",
    "takes",
    "took",
    "put",
    "puts",
    "let",
    "lets",
    "keep",
    "keeps",
    "kept",
    "begin",
    "begins",
    "began",
    "seem",
    "seems",
    "seemed",
    "help",
    "helps",
    "helped",
    "show",
    "shows",
    "showed",
    "hear",
    "hears",
    "heard",
    "play",
    "plays",
    "played",
    "run",
    "runs",
    "ran",
    "move",
    "moves",
    "moved",
    "live",
    "lives",
    "lived",
    "believe",
    "believes",
    "believed",
    "hold",
    "holds",
    "held",
    "bring",
    "brings",
    "brought",
    "happen",
    "happens",
    "happened",
    "write",
    "writes",
    "wrote",
    "provide",
    "provides",
    "provided",
    "sit",
    "sits",
    "sat",
    "stand",
    "stands",
    "stood",
    "lose",
    "loses",
    "lost",
    "pay",
    "pays",
    "paid",
    "meet",
    "meets",
    "met",
    "include",
    "includes",
    "included",
    "continue",
    "continues",
    "continued",
    "set",
    "sets",
    "learn",
    "learns",
    "learned",
    "change",
    "changes",
    "changed",
    "lead",
    "leads",
    "led",
    "understand",
    "understands",
    "understood",
    "watch",
    "watches",
    "watched",
    "follow",
    "follows",
    "followed",
    "stop",
    "stops",
    "stopped",
    "create",
    "creates",
    "created",
    "speak",
    "speaks",
    "spoke",
    "read",
    "reads",
    "allow",
    "allows",
    "allowed",
    "add",
    "adds",
    "added",
    "spend",
    "spends",
    "spent",
    "grow",
    "grows",
    "grew",
    "open",
    "opens",
    "opened",
    "walk",
    "walks",
    "walked",
    "win",
    "wins",
    "won",
    "offer",
    "offers",
    "offered",
    "remember",
    "remembers",
    "remembered",
    "love",
    "loves",
    "loved",
    "consider",
    "considers",
    "considered",
    "appear",
    "appears",
    "appeared",
    "buy",
    "buys",
    "bought",
    "wait",
    "waits",
    "waited",
    "serve",
    "serves",
    "served",
    "die",
    "dies",
    "died",
    "send",
    "sends",
    "sent",
    "expect",
    "expects",
    "expected",
    "build",
    "builds",
    "built",
    "stay",
    "stays",
    "stayed",
    "fall",
    "falls",
    "fell",
    "cut",
    "cuts",
    "reach",
    "reaches",
    "reached",
    "kill",
    "kills",
    "killed",
    "remain",
    "remains",
    "remained",
    # Common nouns
    "time",
    "year",
    "people",
    "way",
    "day",
    "man",
    "woman",
    "child",
    "world",
    "life",
    "hand",
    "part",
    "place",
    "case",
    "week",
    "company",
    "system",
    "program",
    "question",
    "work",
    "government",
    "number",
    "night",
    "point",
    "home",
    "water",
    "room",
    "mother",
    "area",
    "money",
    "story",
    "fact",
    "month",
    "lot",
    "right",
    "study",
    "book",
    "eye",
    "job",
    "word",
    "business",
    "issue",
    "side",
    "kind",
    "head",
    "house",
    "service",
    "friend",
    "father",
    "power",
    "hour",
    "game",
    "line",
    "end",
    "member",
    "law",
    "car",
    "city",
    "community",
    "name",
    "president",
    "team",
    "minute",
    "idea",
    "kid",
    "body",
    "information",
    "back",
    "parent",
    "face",
    "others",
    "level",
    "office",
    "door",
    "health",
    "person",
    "art",
    "war",
    "history",
    "party",
    "result",
    "change",
    "morning",
    "reason",
    "research",
    "girl",
    "guy",
    "moment",
    "air",
    "teacher",
    "force",
    "education",
    # Common adjectives
    "good",
    "new",
    "first",
    "last",
    "long",
    "great",
    "little",
    "own",
    "other",
    "old",
    "right",
    "big",
    "high",
    "different",
    "small",
    "large",
    "next",
    "early",
    "young",
    "important",
    "few",
    "public",
    "bad",
    "same",
    "able",
    "human",
    "local",
    "sure",
    "free",
    "better",
    "best",
    "true",
    "whole",
    "real",
    "full",
    "hard",
    "possible",
    "special",
    "easy",
    "clear",
    "recent",
    "certain",
    "personal",
    "open",
    "red",
    "difficult",
    "available",
    "likely",
    "short",
    "single",
    "medical",
    "current",
    "wrong",
    "private",
    "past",
    "foreign",
    "fine",
    "common",
    "poor",
    "natural",
    "significant",
    "similar",
    "hot",
    "dead",
    "central",
    "happy",
    "serious",
    "ready",
    "simple",
    "left",
    "physical",
    "general",
    "environmental",
    "financial",
    "blue",
    "democratic",
    "dark",
    "various",
    "entire",
    "close",
    "legal",
    "religious",
    "cold",
    "final",
    "main",
    "green",
    "nice",
    "huge",
    "popular",
    "traditional",
    "cultural",
    # Common adverbs
    "up",
    "so",
    "out",
    "just",
    "now",
    "how",
    "then",
    "more",
    "also",
    "here",
    "well",
    "only",
    "very",
    "even",
    "back",
    "there",
    "down",
    "still",
    "where",
    "too",
    "when",
    "never",
    "today",
    "way",
    "really",
    "most",
    "already",
    "always",
    "often",
    "however",
    "together",
    "likely",
    "simply",
    "generally",
    "instead",
    "actually",
    "ever",
    "ago",
    "far",
    "away",
    "again",
    "rather",
    "almost",
    "especially",
    "later",
    "yet",
    "perhaps",
    "certainly",
    "across",
    "once",
    "probably",
    "sometimes",
    "thus",
    "maybe",
    "indeed",
    "soon",
    "usually",
    "quickly",
    "finally",
    "nearly",
    "suddenly",
    "directly",
    "slowly",
    "exactly",
    "alone",
    # Conjunctions and others
    "and",
    "or",
    "but",
    "if",
    "because",
    "as",
    "than",
    "so",
    "while",
    "although",
    "whether",
    "though",
    "since",
    "until",
    "unless",
    "not",
    "no",
    "yes",
    "all",
    "some",
    "any",
    "many",
    "much",
    "more",
    "most",
    "each",
    "every",
    "both",
    "few",
    "several",
    "enough",
}

# Academic Word List (AWL) - B2+ level vocabulary
# Based on Coxhead (2000) - subset of most common academic words
ACADEMIC_WORDS = {
    "analyze",
    "analysis",
    "analytical",
    "approach",
    "area",
    "assess",
    "assessment",
    "assume",
    "assumption",
    "authority",
    "available",
    "benefit",
    "concept",
    "conceptual",
    "conclude",
    "conclusion",
    "consistent",
    "constitute",
    "context",
    "contract",
    "create",
    "creation",
    "creative",
    "data",
    "define",
    "definition",
    "derive",
    "derived",
    "distribute",
    "distribution",
    "economic",
    "economy",
    "environment",
    "environmental",
    "establish",
    "established",
    "estimate",
    "estimation",
    "evaluate",
    "evaluation",
    "evidence",
    "export",
    "factor",
    "factors",
    "finance",
    "financial",
    "formula",
    "function",
    "functional",
    "identify",
    "identification",
    "income",
    "indicate",
    "indication",
    "individual",
    "individually",
    "interpret",
    "interpretation",
    "involve",
    "involved",
    "involvement",
    "issue",
    "issues",
    "labor",
    "legal",
    "legislate",
    "legislation",
    "major",
    "method",
    "methodology",
    "occur",
    "occurrence",
    "percent",
    "percentage",
    "period",
    "policy",
    "policies",
    "principle",
    "principles",
    "procedure",
    "procedures",
    "process",
    "processes",
    "require",
    "required",
    "requirement",
    "research",
    "researcher",
    "respond",
    "response",
    "role",
    "section",
    "sector",
    "significant",
    "significantly",
    "similar",
    "similarly",
    "source",
    "sources",
    "specific",
    "specifically",
    "structure",
    "structural",
    "theory",
    "theoretical",
    "variable",
    "variables",
    "achieve",
    "achievement",
    "acquire",
    "acquisition",
    "adapt",
    "adaptation",
    "adequate",
    "adequately",
    "adjust",
    "adjustment",
    "administrate",
    "administration",
    "adult",
    "advocate",
    "affect",
    "affected",
    "aggregate",
    "aid",
    "albeit",
    "allocate",
    "allocation",
    "alter",
    "alternative",
    "alternatively",
    "ambiguous",
    "amend",
    "amendment",
    "analogy",
    "annual",
    "annually",
    "anticipate",
    "anticipation",
    "apparent",
    "apparently",
    "append",
    "appendix",
    "appreciate",
    "appreciation",
    "appropriate",
    "appropriately",
    "approximate",
    "approximately",
    "arbitrary",
    "aspect",
    "assemble",
    "assembly",
    "assign",
    "assignment",
    "assist",
    "assistance",
    "attach",
    "attachment",
    "attain",
    "attainment",
    "attitude",
    "attribute",
    "author",
    "automate",
    "automatic",
    "automatically",
    "behalf",
    "bias",
    "bond",
    "brief",
    "briefly",
    "bulk",
    "capable",
    "capacity",
    "category",
    "cease",
    "challenge",
    "channel",
    "chapter",
    "chart",
    "chemical",
    "circumstance",
    "cite",
    "citation",
    "civil",
    "clarify",
    "clarification",
    "classic",
    "classical",
    "clause",
    "code",
    "coherent",
    "coherence",
    "coincide",
    "collapse",
    "colleague",
    "commence",
    "comment",
    "commentary",
    "commission",
    "commit",
    "commitment",
    "commodity",
    "communicate",
    "communication",
    "community",
    "compatible",
    "compensate",
    "compensation",
    "compile",
    "complement",
    "complementary",
    "complex",
    "complexity",
    "component",
    "compound",
    "comprehensive",
    "comprise",
    "compute",
    "computer",
    "conceive",
    "concentrate",
    "concentration",
    "conduct",
    "confer",
    "conference",
    "confine",
    "confirm",
    "confirmation",
    "conflict",
    "conform",
    "conformity",
    "consent",
    "consequent",
    "consequently",
    "considerable",
    "considerably",
    "consist",
    "constant",
    "constantly",
    "constrain",
    "constraint",
    "construct",
    "construction",
    "consult",
    "consultant",
    "consume",
    "consumer",
    "consumption",
    "contact",
    "contemporary",
    "contradict",
    "contradiction",
    "contrary",
    "contrast",
    "contribute",
    "contribution",
    "controversy",
    "controversial",
    "convene",
    "convention",
    "conventional",
    "conversely",
    "convert",
    "conversion",
    "convince",
    "cooperate",
    "cooperation",
    "coordinate",
    "coordination",
    "core",
    "corporate",
    "corporation",
    "correspond",
    "correspondence",
    "corresponding",
    "couple",
    "criteria",
    "criterion",
    "crucial",
    "culture",
    "cultural",
    "currency",
    "cycle",
    "debate",
    "decade",
    "decline",
    "deduce",
    "deduction",
    "definite",
    "definitely",
    "demonstrate",
    "demonstration",
    "denote",
    "deny",
    "denial",
    "depress",
    "depression",
    "despite",
    "detect",
    "detection",
    "deviate",
    "deviation",
    "device",
    "devote",
    "devoted",
    "differentiate",
    "dimension",
    "diminish",
    "discrete",
    "discriminate",
    "discrimination",
    "displace",
    "displacement",
    "display",
    "dispose",
    "disposal",
    "distinct",
    "distinction",
    "distort",
    "distortion",
    "diverse",
    "diversity",
    "document",
    "documentation",
    "domain",
    "domestic",
    "dominate",
    "dominant",
    "draft",
    "drama",
    "dramatic",
    "dramatically",
    "duration",
    "dynamic",
    "edit",
    "edition",
    "editor",
    "element",
    "eliminate",
    "elimination",
    "emerge",
    "emergence",
    "emphasis",
    "emphasize",
    "empirical",
    "enable",
    "encounter",
    "energy",
    "enforce",
    "enforcement",
    "enhance",
    "enhancement",
    "enormous",
    "ensure",
    "entity",
    "equate",
    "equation",
    "equip",
    "equipment",
    "equivalent",
    "erode",
    "erosion",
    "error",
    "ethical",
    "ethics",
    "ethnic",
    "eventual",
    "eventually",
    "evident",
    "evidently",
    "evolve",
    "evolution",
    "exceed",
    "exclude",
    "exclusion",
    "exclusive",
    "exclusively",
    "exhibit",
    "exhibition",
    "expand",
    "expansion",
    "expert",
    "expertise",
    "explicit",
    "explicitly",
    "exploit",
    "exploitation",
    "explore",
    "exploration",
    "expose",
    "exposure",
    "external",
    "extract",
    "extraction",
    "facilitate",
    "facility",
    "feature",
    "federal",
    "fee",
    "file",
    "final",
    "finally",
    "finite",
    "flexible",
    "flexibility",
    "fluctuate",
    "fluctuation",
    "focus",
    "format",
    "formulate",
    "formulation",
    "forthcoming",
    "foundation",
    "found",
    "founder",
    "framework",
    "fundamental",
    "fundamentally",
    "fund",
    "funding",
    "furthermore",
    "gender",
    "generate",
    "generation",
    "globe",
    "global",
    "globally",
    "goal",
    "grade",
    "grant",
    "guarantee",
    "guideline",
    "hence",
    "hierarchy",
    "hierarchical",
    "highlight",
    "hypothesis",
    "hypothetical",
    "identical",
    "ideology",
    "ignorance",
    "ignore",
    "illustrate",
    "illustration",
    "image",
    "immigrate",
    "immigration",
    "impact",
    "implement",
    "implementation",
    "implicate",
    "implication",
    "implicit",
    "implicitly",
    "imply",
    "impose",
    "incentive",
    "incidence",
    "incline",
    "inclination",
    "incorporate",
    "index",
    "indicate",
    "indication",
    "indicator",
    "induce",
    "inevitable",
    "inevitably",
    "infer",
    "inference",
    "infrastructure",
    "inherent",
    "inherently",
    "inhibit",
    "initial",
    "initially",
    "initiate",
    "initiative",
    "injure",
    "injury",
    "innovate",
    "innovation",
    "input",
    "insert",
    "insight",
    "inspect",
    "inspection",
    "instance",
    "institute",
    "institution",
    "institutional",
    "instruct",
    "instruction",
    "instructor",
    "instrument",
    "integral",
    "integrate",
    "integration",
    "integrity",
    "intelligence",
    "intelligent",
    "intense",
    "intensity",
    "intensive",
    "interact",
    "interaction",
    "intermediate",
    "internal",
    "internally",
    "interval",
    "intervene",
    "intervention",
    "intrinsic",
    "invest",
    "investigate",
    "investigation",
    "investment",
    "invoke",
    "isolate",
    "isolation",
    "journal",
    "justify",
    "justification",
    "label",
    "layer",
    "lecture",
    "lecturer",
    "levy",
    "liberal",
    "liberate",
    "liberation",
    "license",
    "likewise",
    "link",
    "locate",
    "location",
    "logic",
    "logical",
    "logically",
    "maintain",
    "maintenance",
    "manipulate",
    "manipulation",
    "manual",
    "manually",
    "margin",
    "marginal",
    "mature",
    "maturity",
    "maximize",
    "maximum",
    "mechanism",
    "media",
    "mediate",
    "mediation",
    "medium",
    "mental",
    "mentally",
    "migrate",
    "migration",
    "military",
    "minimal",
    "minimize",
    "minimum",
    "ministry",
    "minor",
    "minority",
    "mode",
    "modify",
    "modification",
    "monitor",
    "motive",
    "motivation",
    "mutual",
    "mutually",
    "negate",
    "negative",
    "negatively",
    "network",
    "neutral",
    "neutrality",
    "nevertheless",
    "nonetheless",
    "norm",
    "normal",
    "normally",
    "notion",
    "notwithstanding",
    "nuclear",
    "objective",
    "objectively",
    "obtain",
    "obvious",
    "obviously",
    "occupy",
    "occupation",
    "odd",
    "offset",
    "ongoing",
    "option",
    "optional",
    "orient",
    "orientation",
    "outcome",
    "output",
    "overall",
    "overlap",
    "overseas",
    "panel",
    "paradigm",
    "paragraph",
    "parallel",
    "parameter",
    "participate",
    "participation",
    "partner",
    "partnership",
    "passive",
    "perceive",
    "perception",
    "persist",
    "persistence",
    "persistent",
    "perspective",
    "phase",
    "phenomenon",
    "philosophy",
    "philosophical",
    "physical",
    "physically",
    "plus",
    "portion",
    "pose",
    "positive",
    "positively",
    "potential",
    "potentially",
    "practitioner",
    "precede",
    "preceding",
    "precise",
    "precisely",
    "precision",
    "predict",
    "prediction",
    "predominant",
    "predominantly",
    "preliminary",
    "premise",
    "presume",
    "presumably",
    "previous",
    "previously",
    "primary",
    "primarily",
    "prime",
    "principal",
    "principally",
    "prior",
    "priority",
    "proceed",
    "proceeding",
    "professional",
    "professionally",
    "prohibit",
    "prohibition",
    "project",
    "projection",
    "promote",
    "promotion",
    "proportion",
    "proportional",
    "prospect",
    "protocol",
    "psychology",
    "psychological",
    "publication",
    "publish",
    "publisher",
    "purchase",
    "pursue",
    "pursuit",
    "qualitative",
    "qualitatively",
    "quote",
    "quotation",
    "radical",
    "radically",
    "random",
    "randomly",
    "range",
    "ratio",
    "rational",
    "rationale",
    "react",
    "reaction",
    "recover",
    "recovery",
    "refine",
    "refinement",
    "regime",
    "region",
    "regional",
    "register",
    "registration",
    "regulate",
    "regulation",
    "reinforce",
    "reinforcement",
    "reject",
    "rejection",
    "relax",
    "relaxation",
    "release",
    "relevant",
    "relevance",
    "reluctant",
    "reluctance",
    "rely",
    "reliance",
    "remove",
    "removal",
    "render",
    "resolve",
    "resolution",
    "resource",
    "resources",
    "restore",
    "restoration",
    "restrain",
    "restraint",
    "restrict",
    "restriction",
    "retain",
    "retention",
    "reveal",
    "revelation",
    "revenue",
    "reverse",
    "reversal",
    "revise",
    "revision",
    "revolution",
    "revolutionary",
    "rigid",
    "rigidity",
    "scenario",
    "schedule",
    "scheme",
    "scope",
    "seek",
    "select",
    "selection",
    "sequence",
    "sequential",
    "series",
    "sex",
    "sexual",
    "shift",
    "simulate",
    "simulation",
    "site",
    "so-called",
    "sole",
    "solely",
    "somewhat",
    "sophisticated",
    "sophistication",
    "specify",
    "specification",
    "sphere",
    "stable",
    "stability",
    "statistic",
    "statistical",
    "statistically",
    "status",
    "straightforward",
    "strategy",
    "strategic",
    "strategically",
    "stress",
    "strict",
    "strictly",
    "style",
    "submit",
    "submission",
    "subordinate",
    "subsequent",
    "subsequently",
    "subsidy",
    "subsidize",
    "substitute",
    "substitution",
    "successor",
    "sufficient",
    "sufficiently",
    "sum",
    "summary",
    "summarize",
    "supplement",
    "supplementary",
    "survey",
    "survive",
    "survival",
    "suspend",
    "suspension",
    "sustain",
    "sustainable",
    "symbol",
    "symbolic",
    "tape",
    "target",
    "task",
    "team",
    "technical",
    "technically",
    "technique",
    "technology",
    "technological",
    "temporary",
    "temporarily",
    "tense",
    "tension",
    "terminate",
    "termination",
    "text",
    "textual",
    "theme",
    "thereby",
    "thesis",
    "topic",
    "trace",
    "tradition",
    "traditional",
    "traditionally",
    "transfer",
    "transform",
    "transformation",
    "transit",
    "transition",
    "transmit",
    "transmission",
    "transport",
    "transportation",
    "trend",
    "trigger",
    "ultimate",
    "ultimately",
    "undergo",
    "underlie",
    "underlying",
    "undertake",
    "uniform",
    "uniformly",
    "unify",
    "unique",
    "uniquely",
    "utilize",
    "utilization",
    "valid",
    "validity",
    "vary",
    "variation",
    "vehicle",
    "version",
    "via",
    "violate",
    "violation",
    "virtual",
    "virtually",
    "visible",
    "visibility",
    "vision",
    "visual",
    "visually",
    "volume",
    "voluntary",
    "volunteer",
    "welfare",
    "whereas",
    "whereby",
    "widespread",
}

# Function words (grammatical, not content)
FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "because",
    "as",
    "than",
    "so",
    "while",
    "although",
    "whether",
    "though",
    "since",
    "until",
    "unless",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "who",
    "what",
    "which",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "of",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "over",
    "out",
    "up",
    "down",
    "off",
    "away",
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "shall",
    "not",
    "no",
    "yes",
    "all",
    "some",
    "any",
    "many",
    "much",
    "more",
    "most",
    "each",
    "every",
    "both",
    "few",
    "several",
    "enough",
    "other",
    "another",
    "such",
    "own",
    "same",
    "different",
    "just",
    "only",
    "very",
    "even",
    "also",
    "too",
    "still",
    "already",
    "yet",
    "again",
    "always",
    "never",
    "often",
    "sometimes",
    "usually",
    "here",
    "there",
    "now",
    "then",
}


@dataclass
class RangeFeatures:
    """Extracted vocabulary/range features from text."""

    # Lexical diversity
    total_words: int = 0
    unique_words: int = 0
    type_token_ratio: float = 0.0  # TTR
    root_ttr: float = 0.0  # Guiraud's R
    corrected_ttr: float = 0.0  # CTTR

    # Word sophistication
    common_word_ratio: float = 0.0  # % of common/basic words
    academic_word_ratio: float = 0.0  # % of academic words
    rare_word_ratio: float = 0.0  # % of uncommon words
    academic_word_count: int = 0

    # Vocabulary breadth
    avg_word_length: float = 0.0
    long_word_ratio: float = 0.0  # Words > 6 chars
    avg_syllables: float = 0.0

    # Lexical density
    content_word_ratio: float = 0.0  # Content vs function words
    lexical_density: float = 0.0

    # Word frequency distribution
    hapax_legomena: int = 0  # Words appearing once
    hapax_ratio: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RangeScore:
    """CEFR range/vocabulary assessment result."""

    level: str  # A1, A2, B1, B2+
    score: float  # Overall score (0-100)
    confidence: float  # Confidence in assessment

    sub_scores: Dict[str, float] = field(default_factory=dict)
    features: RangeFeatures = field(default_factory=RangeFeatures)
    feedback: List[str] = field(default_factory=list)

    # Sample vocabulary
    academic_words_used: List[str] = field(default_factory=list)
    rare_words_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "level": self.level,
            "score": round(self.score, 1),
            "confidence": round(self.confidence, 3),
            "sub_scores": {k: round(v, 1) for k, v in self.sub_scores.items()},
            "features": self.features.to_dict(),
            "feedback": self.feedback,
            "academic_words_used": self.academic_words_used[:10],
            "rare_words_used": self.rare_words_used[:10],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Range Level: {self.level} (score: {self.score:.1f}/100)",
            f"Confidence: {self.confidence:.1%}",
            "",
            "Sub-scores:",
        ]
        for name, score in self.sub_scores.items():
            lines.append(f"  {name:20}: {score:.1f}/100")

        if self.academic_words_used:
            lines.extend(
                ["", f"Academic words used: {', '.join(self.academic_words_used[:5])}"]
            )

        if self.rare_words_used:
            lines.extend(
                [f"Rare/advanced words: {', '.join(self.rare_words_used[:5])}"]
            )

        if self.feedback:
            lines.extend(["", "Feedback:"])
            for fb in self.feedback:
                lines.append(f"  - {fb}")

        return "\n".join(lines)


class RangeFeatureExtractor:
    """
    Extract vocabulary/range features from text.

    Example:
        >>> extractor = RangeFeatureExtractor()
        >>> features = extractor.extract("I think learning English is important...")
        >>> print(features.type_token_ratio, features.academic_word_ratio)
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self.common_words = COMMON_WORDS
        self.academic_words = ACADEMIC_WORDS
        self.function_words = FUNCTION_WORDS

    def extract(self, text: str) -> Tuple[RangeFeatures, List[str], List[str]]:
        """
        Extract range features from text.

        Uses pure calculation functions from calculations.py.

        Args:
            text: Transcription text

        Returns:
            Tuple of (RangeFeatures, academic_words_found, rare_words_found)
        """
        if not text or not text.strip():
            return RangeFeatures(), [], []

        # Tokenize (using calculations.py)
        words = tokenize(text)

        if not words:
            return RangeFeatures(), [], []

        # Basic counts (using calculations.py)
        total_words = len(words)
        word_freq = get_word_frequencies(words)
        unique_words = len(word_freq)

        # 1. LEXICAL DIVERSITY (using calculations.py)
        ttr = calculate_ttr(total_words, unique_words)
        root_ttr = calculate_root_ttr(total_words, unique_words)
        cttr = calculate_corrected_ttr(total_words, unique_words)

        # 2. WORD SOPHISTICATION
        common_count = sum(1 for w in words if w in self.common_words)
        academic_found = [w for w in words if w in self.academic_words]
        academic_count = len(academic_found)

        # Rare words = not common and not academic
        rare_found = [
            w
            for w in set(words)
            if w not in self.common_words
            and w not in self.academic_words
            and len(w) > 3  # Exclude very short words
        ]
        rare_count = sum(1 for w in words if w in rare_found)

        common_ratio = common_count / total_words if total_words > 0 else 0
        academic_ratio = academic_count / total_words if total_words > 0 else 0
        rare_ratio = rare_count / total_words if total_words > 0 else 0

        # 3. VOCABULARY BREADTH (using calculations.py)
        avg_word_len = calculate_avg_word_length(words)
        long_word_rat = calculate_long_word_ratio(words, threshold=6)
        syllable_counts = [count_syllables(w) for w in words]
        avg_syllables = (
            sum(syllable_counts) / len(syllable_counts) if syllable_counts else 0
        )

        # 4. LEXICAL DENSITY
        content_words = [w for w in words if w not in self.function_words]
        content_ratio = len(content_words) / total_words if total_words > 0 else 0
        lex_density = (
            len(set(content_words)) / len(content_words) if content_words else 0
        )

        # 5. WORD FREQUENCY DISTRIBUTION (using calculations.py)
        hapax, hapax_rat = calculate_hapax_ratio(word_freq)

        features = RangeFeatures(
            # Diversity
            total_words=total_words,
            unique_words=unique_words,
            type_token_ratio=ttr,
            root_ttr=root_ttr,
            corrected_ttr=cttr,
            # Sophistication
            common_word_ratio=common_ratio,
            academic_word_ratio=academic_ratio,
            rare_word_ratio=rare_ratio,
            academic_word_count=academic_count,
            # Breadth
            avg_word_length=avg_word_len,
            long_word_ratio=long_word_rat,
            avg_syllables=avg_syllables,
            # Density
            content_word_ratio=content_ratio,
            lexical_density=lex_density,
            # Distribution
            hapax_legomena=hapax,
            hapax_ratio=hapax_rat,
        )

        return features, list(set(academic_found)), rare_found


class RangeScorer:
    """
    Convert range features to CEFR level using research-based thresholds.

    Scoring dimensions:
    1. Lexical Diversity (30%) - TTR, unique words
    2. Word Sophistication (35%) - Academic/rare words
    3. Vocabulary Breadth (20%) - Word length, complexity
    4. Lexical Density (15%) - Content word usage

    Example:
        >>> scorer = RangeScorer()
        >>> features = RangeFeatures(type_token_ratio=0.6, academic_word_ratio=0.05, ...)
        >>> result = scorer.score(features, [], [])
        >>> print(result.level)  # "B1"
    """

    DEFAULT_WEIGHTS = {
        "diversity": 0.30,
        "sophistication": 0.35,
        "breadth": 0.20,
        "density": 0.15,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def score(
        self,
        features: RangeFeatures,
        academic_words: List[str],
        rare_words: List[str],
    ) -> RangeScore:
        """
        Score range features and return CEFR level.

        Args:
            features: Extracted range features
            academic_words: List of academic words found
            rare_words: List of rare words found

        Returns:
            RangeScore with level, confidence, and breakdown
        """
        # Calculate sub-scores (0-100)
        diversity_score = self._score_diversity(features)
        sophistication_score = self._score_sophistication(features)
        breadth_score = self._score_breadth(features)
        density_score = self._score_density(features)

        sub_scores = {
            "diversity": diversity_score,
            "sophistication": sophistication_score,
            "breadth": breadth_score,
            "density": density_score,
        }

        # Weighted overall score
        overall_score = sum(self.weights[dim] * sub_scores[dim] for dim in self.weights)

        # Map to CEFR level
        level = self._score_to_level(overall_score)

        # Calculate confidence
        confidence = self._calculate_confidence(overall_score, features)

        # Generate feedback
        feedback = self._generate_feedback(features, sub_scores, level)

        return RangeScore(
            level=level,
            score=overall_score,
            confidence=confidence,
            sub_scores=sub_scores,
            features=features,
            feedback=feedback,
            academic_words_used=academic_words,
            rare_words_used=rare_words,
        )

    def _score_diversity(self, features: RangeFeatures) -> float:
        """
        Score lexical diversity (0-100).

        Based on TTR benchmarks:
        - A1: TTR < 0.40
        - A2: TTR 0.40-0.50
        - B1: TTR 0.50-0.60
        - B2+: TTR > 0.60
        """
        # TTR score (0-50) - note: TTR decreases with text length
        # Use corrected TTR for longer texts
        ttr = (
            features.corrected_ttr
            if features.total_words > 100
            else features.type_token_ratio
        )

        if ttr < 0.30:
            ttr_score = 10
        elif ttr < 0.40:
            ttr_score = 10 + (ttr - 0.30) * 200  # 10-30
        elif ttr < 0.50:
            ttr_score = 30 + (ttr - 0.40) * 200  # 30-50
        elif ttr < 0.60:
            ttr_score = 50 + (ttr - 0.50) * 200  # 50-70
        elif ttr < 0.70:
            ttr_score = 70 + (ttr - 0.60) * 200  # 70-90
        else:
            ttr_score = 90 + min(10, (ttr - 0.70) * 100)  # 90-100

        # Unique words bonus (0-30)
        unique = features.unique_words
        if unique < 20:
            unique_score = 5
        elif unique < 50:
            unique_score = 5 + (unique - 20) * 0.5  # 5-20
        elif unique < 100:
            unique_score = 20 + (unique - 50) * 0.2  # 20-30
        else:
            unique_score = 30

        # Hapax ratio (0-20) - higher = more diverse
        hapax = features.hapax_ratio
        if hapax < 0.3:
            hapax_score = 5
        elif hapax < 0.5:
            hapax_score = 5 + (hapax - 0.3) * 50  # 5-15
        else:
            hapax_score = 15 + min(5, (hapax - 0.5) * 10)  # 15-20

        return min(100, ttr_score * 0.5 + unique_score + hapax_score)

    def _score_sophistication(self, features: RangeFeatures) -> float:
        """
        Score word sophistication (0-100).

        Based on academic/rare word usage:
        - A1: < 1% academic words
        - A2: 1-2% academic words
        - B1: 2-4% academic words
        - B2+: > 4% academic words
        """
        # Academic word score (0-50)
        academic = features.academic_word_ratio

        if academic < 0.01:
            academic_score = 10
        elif academic < 0.02:
            academic_score = 10 + (academic - 0.01) * 2000  # 10-30
        elif academic < 0.04:
            academic_score = 30 + (academic - 0.02) * 1000  # 30-50
        elif academic < 0.06:
            academic_score = 50 + (academic - 0.04) * 1000  # 50-70
        elif academic < 0.08:
            academic_score = 70 + (academic - 0.06) * 1000  # 70-90
        else:
            academic_score = 90 + min(10, (academic - 0.08) * 500)  # 90-100

        # Rare word score (0-30)
        rare = features.rare_word_ratio

        if rare < 0.05:
            rare_score = 5
        elif rare < 0.10:
            rare_score = 5 + (rare - 0.05) * 200  # 5-15
        elif rare < 0.15:
            rare_score = 15 + (rare - 0.10) * 200  # 15-25
        else:
            rare_score = 25 + min(5, (rare - 0.15) * 50)  # 25-30

        # Common word penalty (0-20) - lower common ratio = higher score
        common = features.common_word_ratio

        if common > 0.90:
            common_score = 5
        elif common > 0.80:
            common_score = 5 + (0.90 - common) * 100  # 5-15
        elif common > 0.70:
            common_score = 15 + (0.80 - common) * 50  # 15-20
        else:
            common_score = 20

        return min(100, academic_score * 0.5 + rare_score + common_score)

    def _score_breadth(self, features: RangeFeatures) -> float:
        """
        Score vocabulary breadth (0-100).

        Based on word complexity:
        - A1: avg length < 4, few long words
        - A2: avg length 4-4.5
        - B1: avg length 4.5-5
        - B2+: avg length > 5, many long words
        """
        # Word length score (0-50)
        avg_len = features.avg_word_length

        if avg_len < 3.5:
            length_score = 15
        elif avg_len < 4.0:
            length_score = 15 + (avg_len - 3.5) * 40  # 15-35
        elif avg_len < 4.5:
            length_score = 35 + (avg_len - 4.0) * 40  # 35-55
        elif avg_len < 5.0:
            length_score = 55 + (avg_len - 4.5) * 40  # 55-75
        elif avg_len < 5.5:
            length_score = 75 + (avg_len - 5.0) * 40  # 75-95
        else:
            length_score = 95 + min(5, (avg_len - 5.5) * 10)  # 95-100

        # Long word ratio score (0-30)
        long_ratio = features.long_word_ratio

        if long_ratio < 0.05:
            long_score = 5
        elif long_ratio < 0.10:
            long_score = 5 + (long_ratio - 0.05) * 200  # 5-15
        elif long_ratio < 0.15:
            long_score = 15 + (long_ratio - 0.10) * 200  # 15-25
        else:
            long_score = 25 + min(5, (long_ratio - 0.15) * 50)  # 25-30

        # Syllable complexity (0-20)
        avg_syl = features.avg_syllables

        if avg_syl < 1.2:
            syl_score = 5
        elif avg_syl < 1.4:
            syl_score = 5 + (avg_syl - 1.2) * 50  # 5-15
        elif avg_syl < 1.6:
            syl_score = 15 + (avg_syl - 1.4) * 25  # 15-20
        else:
            syl_score = 20

        return min(100, length_score * 0.5 + long_score + syl_score)

    def _score_density(self, features: RangeFeatures) -> float:
        """
        Score lexical density (0-100).

        Higher content word ratio = more informative speech
        """
        # Content word ratio (0-60)
        content = features.content_word_ratio

        if content < 0.40:
            content_score = 20
        elif content < 0.50:
            content_score = 20 + (content - 0.40) * 200  # 20-40
        elif content < 0.60:
            content_score = 40 + (content - 0.50) * 200  # 40-60
        else:
            content_score = 60

        # Lexical density (0-40)
        density = features.lexical_density

        if density < 0.40:
            density_score = 10
        elif density < 0.50:
            density_score = 10 + (density - 0.40) * 150  # 10-25
        elif density < 0.60:
            density_score = 25 + (density - 0.50) * 150  # 25-40
        else:
            density_score = 40

        return min(100, content_score + density_score)

    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to CEFR level."""
        return score_to_cefr_level(score)

    def _calculate_confidence(
        self,
        score: float,
        features: RangeFeatures,
    ) -> float:
        """
        Calculate confidence in the assessment.

        Higher confidence when:
        - Score is far from level boundaries
        - Sufficient word count
        """
        # Distance from boundaries
        boundaries = [0, 35, 55, 75, 100]
        min_distance = min(abs(score - b) for b in boundaries)
        boundary_confidence = min(0.4, min_distance / 25)

        # Word count confidence
        words = features.total_words
        if words >= 100:
            word_confidence = 0.3
        elif words >= 50:
            word_confidence = 0.2
        elif words >= 25:
            word_confidence = 0.1
        else:
            word_confidence = 0.05

        # Base confidence
        base_confidence = 0.3

        return min(0.95, base_confidence + boundary_confidence + word_confidence)

    def _generate_feedback(
        self,
        features: RangeFeatures,
        sub_scores: Dict[str, float],
        level: str,
    ) -> List[str]:
        """Generate actionable feedback based on scores."""
        feedback = []

        # Diversity feedback
        if features.type_token_ratio < 0.40:
            feedback.append(
                "Limited vocabulary variety. Try using synonyms and "
                "avoiding word repetition."
            )

        # Sophistication feedback
        if features.academic_word_ratio < 0.02 and level in ["B1", "B2+"]:
            feedback.append(
                "Consider using more academic/formal vocabulary "
                "appropriate for your level."
            )

        if features.common_word_ratio > 0.85:
            feedback.append(
                "Vocabulary is mostly basic words. Try incorporating "
                "more advanced vocabulary."
            )

        # Breadth feedback
        if features.avg_word_length < 4.0:
            feedback.append(
                "Using mostly short, simple words. Practice using "
                "longer, more precise vocabulary."
            )

        # Positive feedback
        if sub_scores["diversity"] >= 70:
            feedback.append("Good vocabulary variety - using diverse words.")

        if sub_scores["sophistication"] >= 70:
            feedback.append("Good use of advanced vocabulary.")

        if features.academic_word_count >= 5:
            feedback.append(
                f"Using academic vocabulary well ({features.academic_word_count} academic words)."
            )

        return feedback


class RangeAssessor:
    """
    Complete range/vocabulary assessment pipeline.

    Combines feature extraction and scoring into a single interface.

    Example:
        >>> assessor = RangeAssessor()
        >>> result = assessor.assess("I think learning English is very important...")
        >>> print(result.level)  # "B1"
        >>> print(result.summary())
    """

    def __init__(
        self,
        language: str = "en",
        weights: Optional[Dict[str, float]] = None,
    ):
        self.extractor = RangeFeatureExtractor(language=language)
        self.scorer = RangeScorer(weights=weights)

    def assess(self, text: str) -> RangeScore:
        """
        Assess vocabulary range from text.

        Args:
            text: Transcription text

        Returns:
            RangeScore with level, confidence, and detailed breakdown
        """
        # Extract features
        features, academic_words, rare_words = self.extractor.extract(text)

        # Score and return
        return self.scorer.score(features, academic_words, rare_words)


# Convenience function
def assess_range(text: str, language: str = "en") -> RangeScore:
    """
    Quick range/vocabulary assessment.

    Args:
        text: Transcription text
        language: Language code (default: "en")

    Returns:
        RangeScore
    """
    assessor = RangeAssessor(language=language)
    return assessor.assess(text)


if __name__ == "__main__":
    # Demo with sample texts
    print("Range Assessment Demo")
    print("=" * 60)

    # A1 level text
    a1_text = """
    I like my house. It is big. I have a dog. My dog is nice.
    I go to school. School is good. I like my teacher.
    """

    # B1 level text
    b1_text = """
    I believe that learning English is very important for my career
    because many international companies require employees to communicate
    effectively in English. Additionally, being able to speak English
    allows me to access more information and connect with people from
    different countries and cultures.
    """

    # B2+ level text
    b2_text = """
    The implementation of sustainable development strategies requires
    a comprehensive understanding of environmental, economic, and social
    factors. Research indicates that organizations which prioritize
    sustainability tend to demonstrate superior long-term performance.
    Furthermore, the integration of renewable energy sources has become
    increasingly significant in addressing climate change challenges.
    """

    assessor = RangeAssessor()

    for name, text in [("A1", a1_text), ("B1", b1_text), ("B2+", b2_text)]:
        print(f"\n{'=' * 60}")
        print(f"Sample: {name} level text")
        print("-" * 60)
        print(f"Text: {text.strip()[:100]}...")
        print("-" * 60)

        result = assessor.assess(text)
        print(result.summary())
