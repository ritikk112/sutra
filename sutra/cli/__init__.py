"""Sutra command-line interface.

The ``sutra`` console script (entry point ``sutra.cli.main:app``) wraps the
existing pipelines / MCP server / frontend and adds the guided ``sutra init``
setup wizard.

Design invariant: prompting (questionary / Rich) is kept strictly separate from
logic (detect / validate / provision / config_io) so every non-interactive
building block is unit-testable without a TTY.
"""
