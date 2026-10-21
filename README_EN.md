# BalloonTranslator

[简体中文](README.md) | **English**

**A Windows tool for translating Japanese manga, with a focus on making translated text easier to read and more natural in comic images.**

- Repository: [shuxuanliang95-create/BalloonTranslator](https://github.com/shuxuanliang95-create/BalloonTranslator)
- Current focus: improve the size, clarity, and visibility of translated text in v12.6 while keeping existing features stable.

## Quick Start: Installation and Usage

Before using the project, follow the [installation guide](INSTALL.md) to prepare Python 3.11, Python dependencies, the local Ollama translation model, and the OCR and LaMa image-inpainting models. **Downloading the repository alone is not enough; the model files must be prepared separately as described in the guide.**

- **Install dependencies:** double-click `install_windows_v12_6.bat`
- **Download the OCR models:** double-click `download_ppocr_manga_models.bat`
- **Download the inpainting model:** double-click `download_manga_inpaint_model.bat`
- **Start the app:** make sure the `hy-mt2-7b` model is available in Ollama, then double-click `start_windows_v12_6.bat`
- **Open the interface:** visit http://127.0.0.1:7860 in a browser on the same computer

Note: the current installation script does not explicitly install PyTorch, but the LaMa inpainting feature requires it. See [INSTALL.md](INSTALL.md) for details. The full setup still needs to be tested on a clean target computer. If you encounter problems, please include logs and environment details.

**Need help?** [Installation guide](INSTALL.md) · [Report a bug](.github/ISSUE_TEMPLATE/bug_report.md) · [Contribution guide](CONTRIBUTING.md)

---

## 1. About the Project

BalloonTranslator is a Windows tool for Japanese manga translation. Its goal is to help users process Japanese text in manga images and display translated text in the comic artwork.

The project involves text recognition, translation, image processing, text rendering, and runtime environment setup. Contributions are welcome from developers interested in Python, OCR, image processing, Japanese translation, and Windows packaging.

> The issues listed below are primarily based on usage feedback. Unless explicitly marked as verified, they still need to be confirmed in a reproducible test environment.

## 2. Current Development Goals

The main goal is to **improve the readability of translated text and overall stability without compromising the current translation results or existing features.**

Areas of interest include:

- Make translated text larger and easier to read in manga images.
- Improve text clarity, contrast, and visibility.
- Improve outlines and readability against complex backgrounds.
- Improve vertical text layout and rendering.
- Investigate rendering glitches, missing areas, and image-inpainting artifacts.
- Preserve the existing batch-processing interface and workflow.
- Improve installation, startup, and dependency setup on Windows.

## 3. Version Feedback and Known Issues

### v12.4: An Important Visual Baseline

Previous user feedback indicated that some text-rendering results in v12.4 looked better than those in later experimental versions. Use v12.4 as an important visual comparison baseline, especially for text positioning, layout, background blending, missing areas, slanted vertical text, and the batch-processing interface.

### v12.5 and Later Experiments: Further Comparison Needed

Previous feedback mentioned unsatisfactory text rendering, local gaps, and unnatural text layout. Compare the same input images with the same settings and processing steps to identify which stage causes each issue.

### v12.6: Current Focus

Feedback indicates that v12.6 produces good overall results, but the translated text can still be too small and insufficiently prominent. Current priorities are:

1. Increase text size appropriately without damaging existing translation results.
2. Improve outlines, contrast, and readability against complex backgrounds.
3. Check rendering consistency for horizontal and vertical text.
4. Avoid introducing new gaps, misalignment, or image artifacts when changing text size.
5. Avoid unnecessary changes to the batch-processing interface and established workflow.

Small, testable, and reversible changes are preferred over rewriting the translation pipeline.

## 4. Areas Where Help Is Welcome

### A. Translated Text Is Too Small or Not Prominent Enough

Help investigate appropriate adjustments to font size, outline thickness, text color, and contrast.

**Desired results:**
- Text is easier to read and has clear edges.
- Larger text does not significantly overflow the original text area.
- Both horizontal and vertical text render correctly.
- Text overlap and obstruction are not noticeably increased.

### B. Missing Text or Rendering Gaps

Some previous test images showed similar issues, but the exact causes have not been confirmed. Potential areas to investigate include text-region detection and cropping, inpainting or background filling, text masks and drawing boundaries, and font outlines and antialiasing.

Please compare output images and intermediate results to locate the problem before making broad code changes.

### C. Unnatural Vertical Text Layout

Help review vertical text rendering, character spacing, line spacing, text direction, and boundary calculations.

**Desired results:**
- Characters are arranged naturally and in the correct direction.
- No obvious slant, misalignment, or clipping.
- Outline quality is consistent with horizontal text.
- Multiple lines or longer text do not noticeably overflow.

### D. Windows Installation or Startup Failures

A previous report described a Python dependency installation failure involving `torch` and `torchvision`. The install command used pinned dependency versions and the Python interpreter from the project's runtime environment.

Areas to investigate:
- Compatibility between Python and dependency versions.
- Compatibility between PyTorch and Torchvision versions.
- Network downloads, package indexes, or platform compatibility.
- Clear error messages and safe retry behavior in installation scripts.

The root cause has not yet been confirmed. Please do not attribute it to a specific dependency version without complete logs. Include the full error, Python version, Windows version, and command used. Remove tokens, secrets, and private paths before sharing logs.

### E. Font Origin and License

The repository contains `NotoSansSC.ttf`, but the exact origin and version of this file have not yet been verified. Help identify its metadata, original source, and applicable license so that accurate copyright and license information can be added.

Until the file's origin is confirmed, do not replace it casually or add unverified copyright information.

### F. Project Structure and Automated Tests

Help check whether dependency declarations are complete, Windows launch scripts match the actual Python environment, test configuration references missing files or environment settings, and tests can be added for text layout, font rendering, and basic startup.

Please describe the finding and proposed fix first. Do not delete files whose purpose has not been confirmed.

## 5. Contribution Principles

1. Read the relevant code and reproduce the issue where possible before changing anything.
2. Prefer small, clearly scoped changes and explain the purpose of each change.
3. Do not casually rewrite OCR, translation, image-inpainting, or text-rendering pipelines.
4. Preserve the existing batch-processing layout and workflow.
5. Compare results using the same test images before and after a change whenever possible.
6. Include before-and-after images for visual changes when practical.
7. Describe changed files, test methods, test results, and known limitations.
8. Clearly label features that have not been tested in practice; do not claim unrun tests passed.

## 6. Reporting an Issue

When reporting a problem, please include as much of the following as possible:

- Version used, such as v12.4 or v12.6.
- Windows and Python versions.
- Steps taken when the issue occurred.
- Expected behavior and actual behavior.
- A reproducible input image or a sanitized example.
- Full error messages or terminal logs.
- For visual issues, both the original image and the processed result.
- If the issue only occurs with certain images, any shared characteristics.

Do not upload API keys, passwords, private manga files, or other content you are not authorized to publish. Check logs for tokens, usernames, personal directories, and other private information before submitting them.

## 7. How to Contribute

You are welcome to report bugs, suggest improvements through GitHub Issues, or submit a Pull Request.

Suggested workflow:

1. Read the relevant code and reproduce the issue.
2. Describe the symptoms, analysis, and proposed change in an Issue.
3. Keep changes focused and independently reviewable where possible.
4. Provide before-and-after results and test records.
5. Wait for the maintainer to review the change before it is merged.

For larger changes, please discuss the approach first to avoid conflicts with the current development direction. See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## 8. Current Project Status

The project's current priorities are Windows usability, translated-text readability, and runtime stability.

**Top priority: make v12.6 translated text larger, clearer, and more prominent while preserving existing features and the batch-processing workflow, and avoid introducing new rendering issues.**

Thanks to everyone willing to test the project, report problems, and contribute improvements.
