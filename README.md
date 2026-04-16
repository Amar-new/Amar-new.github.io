# Amar's Notebook

A personal blog on Machine Learning, Mathematics, and Deep Learning — built with Jekyll and hosted on GitHub Pages.

**Live site:** [https://amar-new.github.io](https://amar-new.github.io)

---

## About Me

I'm Amar — an LLM researcher exploring the mathematical foundations of large language models, deep learning architectures, and scalable AI systems. This notebook is where I write about the ideas I encounter in my research: from the calculus behind backpropagation to the engineering challenges of training models at scale.

I believe the best way to truly understand something is to explain it clearly. Each post here is an attempt to do exactly that.

---

## How This Blog Works

This is a [Jekyll](https://jekyllrb.com/) blog hosted on [GitHub Pages](https://pages.github.com/). Jekyll is a static site generator that converts Markdown files into a complete website. GitHub Pages builds and deploys it automatically every time you push.

**The key benefit:** Writing a new blog post is as simple as creating a `.md` (Markdown) file. No HTML required.

---

## Writing a New Blog Post

### Step 1: Create a Markdown file

Create a new file in the `_posts/` folder with this naming convention:

```
_posts/YYYY-MM-DD-your-post-title.md
```

For example: `_posts/2026-05-01-attention-is-all-you-need.md`

### Step 2: Add front matter

Every post starts with a YAML header between `---` lines:

```yaml
---
layout: post
title: "Your Post Title Here"
date: 2026-05-01
category: ml
category-label: Machine Learning
excerpt: "A one-line summary that appears on the homepage."
---
```

**Category options:**
| category | category-label       | Color       |
|----------|---------------------|-------------|
| `math`   | Mathematics         | Rust red    |
| `ml`     | Machine Learning    | Forest green|
| `code`   | Software Engineering| Slate blue  |

### Step 3: Write your content in Markdown

```markdown
This is a paragraph. You can use **bold**, *italic*, and `inline code`.

## This Is a Section Heading

Here's a list:
- First item
- Second item
- Third item

Here's a code block:

​```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
​```
```

### Step 4: Adding Math (optional)

Use LaTeX syntax for math. Inline math uses single dollar signs, display math uses the `math-block` div:

```markdown
The sigmoid function $\sigma(x)$ maps reals to $(0, 1)$.

<div class="math-block">
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
</div>
```

### Step 5: Push to GitHub

```bash
git add _posts/2026-05-01-attention-is-all-you-need.md
git commit -m "New post: Attention Is All You Need"
git push
```

The site rebuilds automatically in ~1-2 minutes. That's it.

---

## One-Time Setup (if starting fresh)

### Option A: Just push these files (easiest)

1. Clone your repo:
   ```bash
   git clone https://github.com/Amar-new/Amar-new.github.io.git
   cd Amar-new.github.io
   ```

2. Delete all existing files (keep `.git/`):
   ```bash
   git rm -r .
   ```

3. Copy all the blog files into the repo folder (maintaining the folder structure).

4. Push:
   ```bash
   git add .
   git commit -m "Rebuild as Jekyll blog"
   git push origin main
   ```

5. Go to **Settings → Pages** in your GitHub repo. Set:
   - Source: **Deploy from a branch**
   - Branch: **main** / **root**
   - Click **Save**

6. Wait 1–2 minutes. Your blog is live at `https://amar-new.github.io/`.

### Option B: Via GitHub web interface (no git needed)

1. Go to your repo on GitHub
2. Delete old files (`index.html`, `IITPKDLogo.jpg`, `baymax.png`, `common.css`)
3. For each file, click **Add file → Create new file**
4. Type the full path as the filename to create folders automatically:
   - `_config.yml`
   - `Gemfile`
   - `index.html`
   - `about.md`
   - `assets/css/style.css`
   - `_layouts/default.html`
   - `_layouts/post.html`
   - `_includes/head.html`
   - `_includes/header.html`
   - `_includes/footer.html`
   - `_posts/2026-04-16-sigmoid-function.md`
   - `_posts/2026-04-15-gradient-descent.md`
5. Paste the contents of each file
6. Enable GitHub Pages in Settings → Pages

### Optional: Preview locally

If you want to preview posts before pushing, install Jekyll locally:

```bash
# Install Ruby (if not already installed)
# macOS: brew install ruby
# Ubuntu: sudo apt install ruby-full

gem install bundler jekyll
bundle install
bundle exec jekyll serve
```

Then visit `http://localhost:4000`. This is optional — you can always just push and preview on the live site.

---

## File Structure

```
Amar-new.github.io/
├── _config.yml            ← Site settings (name, URL, etc.)
├── Gemfile                ← Ruby dependencies
├── index.html             ← Homepage (auto-lists all posts)
├── about.md               ← About page
├── _layouts/
│   ├── default.html       ← Base HTML template
│   └── post.html          ← Blog post template (includes MathJax)
├── _includes/
│   ├── head.html          ← <head> tag (fonts, CSS, SEO)
│   ├── header.html        ← Site header & navigation
│   └── footer.html        ← Site footer
├── assets/
│   └── css/
│       └── style.css      ← All styles
├── _posts/
│   ├── 2026-04-16-sigmoid-function.md     ← Blog post
│   └── 2026-04-15-gradient-descent.md     ← Blog post
└── README.md              ← This file
```

---

## Customization

- **Site name & tagline:** Edit `title` and `description` in `_config.yml`
- **Author name:** Edit `author` in `_config.yml`
- **Colors:** Edit CSS variables in `assets/css/style.css` under `:root`
- **Add a new category:** Add a `.cat-yourcategory` CSS rule with your color
- **Social links:** Edit `_includes/footer.html` and `about.md`

---

## Tech Stack

- **Jekyll** — static site generator (built into GitHub Pages)
- **MathJax 3** — LaTeX math rendering (loaded from CDN)
- **Google Fonts** — Playfair Display, Source Serif 4, DM Sans, JetBrains Mono
- **No JavaScript frameworks** — pure HTML/CSS with Jekyll templating
