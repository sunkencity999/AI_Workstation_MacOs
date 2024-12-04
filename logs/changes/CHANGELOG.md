# Change Log

All notable changes to the AI Workstation will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.1.0] - 2024-02-14
### Added
- Testing suite with pytest
  - Added conftest.py with mock fixtures
  - Added test_core_functions.py with comprehensive tests
  - Created test file structure
- Change logging system
  - Added CHANGELOG.md
  - Created logs/changes directory for tracking modifications

### Changed
- Updated browser tab title to "AI Workstation"
- Added centered, underlined title to sidebar

### Technical Details
- Test coverage includes:
  - API key loading
  - Shell command execution
  - Python script execution
  - Ollama server status checking
  - Web search functionality
  - File processing (PDF, DOCX)
  - Image processing
