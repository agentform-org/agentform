# Agentform Documentation

This directory contains the documentation for Agentform, built with Jekyll and deployed to GitHub Pages.

## Local Development

To build and preview the documentation locally:

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Then open http://localhost:4000 in your browser.

## Structure

- `index.md` - Homepage
- `getting-started.md` - Installation and first steps
- `examples.md` - Example configurations
- `modules.md` - Module system documentation
- `cli-reference.md` - Complete CLI reference
- `architecture.md` - System architecture overview
- `_config.yml` - Jekyll configuration
- `_layouts/` - HTML layouts
- `assets/` - CSS and other static assets

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch. The deployment is handled by the `.github/workflows/pages.yml` workflow.
