; Sutra — Python tree-sitter query file
;
; Captures all class and function definitions regardless of scope.
; The adapter (adapters/python.py) is responsible for all structural
; reasoning: distinguishing methods from functions, detecting nested
; definitions, and filtering by scope.
;
; Capture name convention: dots used as namespace separator (verified
; to work in tree-sitter 0.25.x — keys come through as "class.def" etc.)

; All class declarations (top-level and nested — adapter filters)
(class_definition) @class.def

; All function definitions including methods, async, and decorated
; (adapter walks ancestors to determine scope and kind)
(function_definition) @func.def

; from x.y import a, b as c
(import_from_statement) @import.from

; import os, import numpy as np
(import_statement) @import.plain
