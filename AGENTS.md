# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is **Ketamine Nightmares** — a static content website for ANZCA exam preparation, hosted via GitHub Pages at `ketaminenightmares.com`. There is **no build system, no package manager, and no application code**. Content is hand-authored `.htm` (Word-exported, cleaned with `tidy-html5`) and `.md` (Markdown navigation/index) files.

### Running locally

Serve the site with any static HTTP server:

```
python3 -m http.server 8000
```

- `.htm` files render as full HTML pages directly in the browser.
- `.md` files display as raw Markdown locally (GitHub Pages renders them via built-in Jekyll; there is no local Jekyll config).

### Linting / validation

The only relevant lint tool is `tidy-html5`:

```
tidy -q -e <file.htm>
```

Warnings about HTML5 attribute validity and missing alt text are expected for legacy Word-exported content — these are not blockers.

### Key caveats

- **No dependencies to install.** There is no `package.json`, `requirements.txt`, or any dependency manifest.
- **No build step.** Deployment is `git push` to the GitHub Pages branch.
- The `CNAME` file maps the repo to `ketaminenightmares.com`.
- Content structure: `pex/` (Primary Exam), `fex/` (Final Exam), `pocus/` (Point-of-Care Ultrasound), `admin/` (About/Copyright).
