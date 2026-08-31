/**
 * Strata Documentation Application (Vanilla JS)
 * Fast, accessible, client-side Markdown rendering, individual symbol pages & search engine.
 */

(function () {
  'use strict';

  const data = window.STRATA_DOCS || { navigation: [], documents: {}, searchIndex: [] };

  // DOM Elements
  const navTreeEl = document.getElementById('nav-tree');
  const docArticleEl = document.getElementById('doc-article');
  const breadcrumbsEl = document.getElementById('breadcrumbs');
  const docPaginationEl = document.getElementById('doc-pagination');
  const tocNavEl = document.getElementById('toc-nav');
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const mobileMenuBtn = document.getElementById('mobile-menu-btn');
  const sidebarEl = document.getElementById('sidebar');

  // Search Modal Elements
  const globalSearchBtn = document.getElementById('global-search-btn');
  const searchModalEl = document.getElementById('search-modal');
  const searchInputEl = document.getElementById('search-input');
  const searchResultsEl = document.getElementById('search-results');
  const searchCloseBtn = document.getElementById('search-close-btn');

  let currentDocId = 'tutorials/quickstart';
  let currentAnchorId = '';
  let searchSelectedIdx = 0;
  let currentSearchResults = [];
  const expandedItems = new Set();

  // ==========================================
  // 1. Theme Management
  // ==========================================
  function initTheme() {
    const savedTheme = localStorage.getItem('strata_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('strata_theme', next);
  }

  themeToggleBtn.addEventListener('click', toggleTheme);

  // ==========================================
  // 2. Navigation Tree
  // ==========================================
  function renderNavTree() {
    let html = '';
    data.navigation.forEach(group => {
      html += `
        <div class="nav-group" data-quadrant="${group.quadrant}">
          <div class="nav-group-header">
            <span class="nav-group-title">${group.title}</span>
          </div>
          <ul class="nav-list">
      `;

      group.items.forEach(item => {
        const isDirectActive = item.id === currentDocId;
        const symbols = item.symbols || [];
        const hasSymbols = symbols.length > 0;
        const isChildActive = hasSymbols && symbols.some(s => s.id === currentDocId);
        const isExpanded = expandedItems.has(item.id) || isDirectActive || isChildActive;

        if (isExpanded) {
          expandedItems.add(item.id);
        }

        html += `
          <li class="nav-item">
            <div class="nav-item-row ${isDirectActive ? 'active-row' : ''}">
              <a href="#${item.id}" class="nav-item-link ${isDirectActive ? 'active' : ''}" data-doc-id="${item.id}">
                ${item.title}
              </a>
              ${hasSymbols ? `
                <button class="tree-toggle-btn ${isExpanded ? 'expanded' : ''}" data-item-id="${item.id}" aria-label="Toggle ${item.title}">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
                </button>
              ` : ''}
            </div>
            ${hasSymbols ? `
              <ul class="nav-sublist ${isExpanded ? 'open' : ''}" data-sublist-id="${item.id}">
                ${symbols.map(s => {
                  const isSymbolActive = s.id === currentDocId;
                  return `
                    <li class="nav-symbol-item">
                      <a href="#${s.id}" class="nav-subitem-link ${isSymbolActive ? 'active' : ''}" data-doc-id="${s.id}" title="${escapeHtml(s.title)}">
                        <span>${s.title}</span>
                        <span class="subitem-badge">${s.kind}</span>
                      </a>
                    </li>

                  `;
                }).join('')}
              </ul>
            ` : ''}
          </li>
        `;
      });

      html += `
          </ul>
        </div>
      `;
    });
    navTreeEl.innerHTML = html;

    // Attach toggle listeners
    document.querySelectorAll('.tree-toggle-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const itemId = btn.getAttribute('data-item-id');
        const sublist = document.querySelector(`ul[data-sublist-id="${itemId}"]`);
        if (sublist) {
          if (sublist.classList.contains('open')) {
            sublist.classList.remove('open');
            btn.classList.remove('expanded');
            expandedItems.delete(itemId);
          } else {
            sublist.classList.add('open');
            btn.classList.add('expanded');
            expandedItems.add(itemId);
          }
        }
      });
    });
  }

  function updateActiveNav() {
    document.querySelectorAll('.nav-item-link, .nav-subitem-link').forEach(link => {
      const docId = link.getAttribute('data-doc-id');
      if (docId === currentDocId) {
        link.classList.add('active');
        const row = link.closest('.nav-item-row');
        if (row) row.classList.add('active-row');
      } else {
        link.classList.remove('active');
        const row = link.closest('.nav-item-row');
        if (row && !row.querySelector('.nav-item-link.active')) row.classList.remove('active-row');
      }
    });
  }

  // ==========================================
  // 3. Single-Pass Syntax Highlighter & Markdown Parser
  // ==========================================
  function escapeHtml(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function highlightMojoCode(code) {
    const tokenRegex = /(#.*$)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\b(?:struct|trait|var|def|fn|alias|raises|in|for|while|if|else|elif|return|from|import|out|mut|copy|borrowed|inout|with|as|pass|break|continue)\b)|(\b(?:Int|Int8|Int16|Int32|Int64|UInt|UInt8|UInt16|UInt32|UInt64|Float16|BFloat16|Float32|Float64|Bool|String|DType|Matrix|MatrixView|CSRMatrix|CSCMatrix|Dataset|List|Scalar|PRNG|DecisionTreeClassifier|DecisionTreeRegressor|RandomForestClassifier|RandomForestRegressor|StandardScaler|MinMaxScaler|RobustScaler|Normalizer|OneHotEncoder|Binarizer|Ridge|LinearRegression|LogisticRegression|PCA|TruncatedSVD|KMeans|MiniBatchKMeans|PipelineRegressor|PipelineClassifier|PipelineTransformer)\b)|(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)/gm;

    let lastIndex = 0;
    let result = '';
    let match;

    while ((match = tokenRegex.exec(code)) !== null) {
      result += escapeHtml(code.slice(lastIndex, match.index));

      if (match[1]) {
        result += `<span class="token-comment">${escapeHtml(match[1])}</span>`;
      } else if (match[2]) {
        result += `<span class="token-string">${escapeHtml(match[2])}</span>`;
      } else if (match[3]) {
        result += `<span class="token-keyword">${escapeHtml(match[3])}</span>`;
      } else if (match[4]) {
        result += `<span class="token-type">${escapeHtml(match[4])}</span>`;
      } else if (match[5]) {
        result += `<span class="token-number">${escapeHtml(match[5])}</span>`;
      }

      lastIndex = tokenRegex.lastIndex;
    }

    result += escapeHtml(code.slice(lastIndex));
    return result;
  }

  function resolveMarkdownLink(href) {
    if (!href) return '#';
    if (href.startsWith('http://') || href.startsWith('https://')) return href;
    if (href.startsWith('#')) {
      if (href.includes('/')) return href;
      return `#${currentDocId}${href}`;
    }
    if (href.startsWith('file:///')) {
      const parts = href.split('/');
      return parts[parts.length - 1];
    }

    // Clean relative path e.g. "Dataset.md", "index.md", "../reference/ensemble/RandomForestClassifier.md"
    let clean = href.replace(/\.md$/, '').replace(/^\.\//, '');
    
    if (clean.startsWith('../')) {
      clean = clean.replace(/^\.\.\//, '');
    } else if (!clean.includes('/')) {
      // Relative to current directory
      const parts = currentDocId.split('/');
      parts.pop(); // remove current file name
      const dir = parts.join('/');
      clean = dir ? `${dir}/${clean}` : clean;
    }

    return `#${clean}`;
  }

  function parseMarkdown(md) {
    if (!md) return '<p>No content available for this document.</p>';

    const lines = md.split('\n');
    let html = [];
    let inCodeBlock = false;
    let codeLanguage = '';
    let codeLines = [];
    let inTable = false;
    let tableRows = [];
    let inList = false;
    let listType = '';

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Code blocks
      if (line.trim().startsWith('```')) {
        if (inCodeBlock) {
          const fullCode = codeLines.join('\n');
          const isHighlightable = codeLanguage === 'mojo' || codeLanguage === 'python';
          const highlighted = isHighlightable ? highlightMojoCode(fullCode) : escapeHtml(fullCode);
          html.push(`
            <div class="code-block-wrapper">
              <div class="code-header">
                <span>${escapeHtml(codeLanguage || 'code')}</span>
                <button class="copy-code-btn" data-code="${encodeURIComponent(fullCode)}">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                  Copy
                </button>
              </div>
              <pre><code>${highlighted}</code></pre>
            </div>
          `);
          inCodeBlock = false;
          codeLines = [];
        } else {
          inCodeBlock = true;
          codeLanguage = line.trim().replace(/^```/, '').trim();
          codeLines = [];
        }
        continue;
      }

      if (inCodeBlock) {
        codeLines.push(line);
        continue;
      }

      // Close list if open and line is not list item
      if (inList && !line.trim().startsWith('- ') && !line.trim().startsWith('* ') && !/^\d+\.\s/.test(line.trim()) && line.trim() !== '') {
        html.push(listType === 'ul' ? '</ul>' : '</ol>');
        inList = false;
      }

      // Table detection - must start and end with pipe, and have delimiter row
      if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
        if (inTable) {
          tableRows.push(line.trim());
          continue;
        } else {
          // Check if next line is a valid table delimiter row (e.g. |:---|---|)
          const nextLine = (i + 1 < lines.length) ? lines[i + 1].trim() : '';
          if (nextLine.startsWith('|') && nextLine.endsWith('|') && /^\|(?:\s*:?-{2,}:?\s*\|)+$/.test(nextLine)) {
            inTable = true;
            tableRows.push(line.trim());
            continue;
          }
        }
      }

      // Close table if open and line is not table row
      if (inTable) {
        html.push(renderTable(tableRows));
        inTable = false;
        tableRows = [];
      }

      // Empty line
      if (line.trim() === '') {
        continue;
      }

      // Block Math Equation ($$ ... $$)
      if (line.trim().startsWith('$$')) {
        const trimmed = line.trim();
        if (trimmed.length > 2 && trimmed.endsWith('$$')) {
          html.push(`<div class="math-block">${trimmed}</div>`);
          continue;
        } else {
          let mathLines = [trimmed];
          let j = i + 1;
          while (j < lines.length) {
            mathLines.push(lines[j]);
            if (lines[j].trim().endsWith('$$')) {
              break;
            }
            j++;
          }
          i = j;
          html.push(`<div class="math-block">${mathLines.join('\n')}</div>`);
          continue;
        }
      }

      // Blockquotes & Callouts
      if (line.trim().startsWith('>')) {
        let bqLines = [];
        let j = i;
        while (j < lines.length && lines[j].trim().startsWith('>')) {
          bqLines.push(lines[j].trim().substring(1).trim());
          j++;
        }
        i = j - 1;

        let firstLine = bqLines[0] || '';
        let calloutMatch = firstLine.match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]/i);
        if (calloutMatch) {
          let type = calloutMatch[1].toLowerCase();
          bqLines.shift();
          html.push(`<div class="callout callout-${type}"><p>${bqLines.map(l => formatInline(l)).join('<br>')}</p></div>`);
        } else if (firstLine.includes('**Overload Note**:') || firstLine.includes('**Overloaded Method**:')) {
          html.push(`<div class="overload-note"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0; color: var(--accent-primary);"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span>${bqLines.map(l => formatInline(l)).join(' ')}</span></div>`);
        } else {
          html.push(`<blockquote><p>${bqLines.map(l => formatInline(l)).join('<br>')}</p></blockquote>`);
        }
        continue;
      }


      // Headings
      if (line.startsWith('# ')) {
        const title = line.substring(2).trim();
        const id = slugify(title);
        html.push(`<h1 id="${id}">${formatInline(title)}</h1>`);
        continue;
      }
      if (line.startsWith('## ')) {
        const title = line.substring(3).trim();
        const id = slugify(title);
        html.push(`<h2 id="${id}">${formatInline(title)}</h2>`);
        continue;
      }
      if (line.startsWith('### ')) {
        const title = line.substring(4).trim();
        const id = slugify(title);
        html.push(`<h3 id="${id}" class="method-title">${formatInline(title)}</h3>`);
        continue;
      }
      if (line.startsWith('#### ')) {
        const title = line.substring(5).trim();
        const id = slugify(title);
        html.push(`<h4 id="${id}">${formatInline(title)}</h4>`);
        continue;
      }

      // Horizontal Rule
      if (line.trim() === '---' || line.trim() === '***') {
        html.push('<hr>');
        continue;
      }

      // Lists
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        if (!inList) {
          inList = true;
          listType = 'ul';
          html.push('<ul>');
        }
        html.push(`<li>${formatInline(line.trim().substring(2))}</li>`);
        continue;
      }

      if (/^\d+\.\s/.test(line.trim())) {
        if (!inList) {
          inList = true;
          listType = 'ol';
          html.push('<ol>');
        }
        const text = line.trim().replace(/^\d+\.\s/, '');
        html.push(`<li>${formatInline(text)}</li>`);
        continue;
      }

      // Paragraph
      html.push(`<p>${formatInline(line.trim())}</p>`);
    }

    if (inList) html.push(listType === 'ul' ? '</ul>' : '</ol>');
    if (inTable) html.push(renderTable(tableRows));

    return html.join('\n');
  }

  function renderTable(rows) {
    if (rows.length < 2) return rows.map(r => `<p>${formatInline(r)}</p>`).join('\n');
    let out = ['<div class="table-wrapper"><table>'];

    
    const headers = rows[0].split('|').map(s => s.trim()).filter((s, idx, arr) => idx > 0 && idx < arr.length - 1);
    out.push('<thead><tr>');
    headers.forEach(h => out.push(`<th>${formatInline(h)}</th>`));
    out.push('</tr></thead><tbody>');

    for (let r = 2; r < rows.length; r++) {
      const cells = rows[r].split('|').map(s => s.trim()).filter((s, idx, arr) => idx > 0 && idx < arr.length - 1);
      out.push('<tr>');
      cells.forEach(c => out.push(`<td>${formatInline(c)}</td>`));
      out.push('</tr>');
    }

    out.push('</tbody></table></div>');
    return out.join('');
  }

  function formatInline(text) {
    if (!text) return '';
    // Temporarily extract inline math to protect LaTeX symbols ($...$)
    const mathTokens = [];
    let sanitized = text.replace(/\$([^\$\n]+)\$/g, (match) => {
      const idx = mathTokens.length;
      mathTokens.push(match);
      return `@@MATH_${idx}@@`;
    });

    sanitized = sanitized
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, text, url) => {
        if (url.startsWith('file:///')) {
          const parts = url.split('/');
          const filename = parts[parts.length - 1];
          return `<span class="file-ref"><code>${filename}</code></span>`;
        }
        const resolvedUrl = resolveMarkdownLink(url);
        return `<a href="${resolvedUrl}">${text}</a>`;
      });

    // Restore inline math
    sanitized = sanitized.replace(/@@MATH_(\d+)@@/g, (_, idx) => mathTokens[parseInt(idx, 10)] || '');
    return sanitized;
  }

  function slugify(text) {
    return text
      .toLowerCase()
      .replace(/[`*]/g, '')
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
  }

  // ==========================================
  // 4. Render Document & TOC
  // ==========================================
  function renderDocument(docId, anchorId) {
    currentDocId = docId;
    currentAnchorId = anchorId || '';

    // Find parent group and item for breadcrumbs
    let currentItem = null;
    let currentGroup = null;
    let parentModule = null;
    let allNavItems = [];

    data.navigation.forEach(group => {
      group.items.forEach(item => {
        allNavItems.push({ ...item, groupTitle: group.title });
        if (item.id === docId) {
          currentItem = item;
          currentGroup = group;
        }
        if (item.symbols) {
          item.symbols.forEach(sym => {
            allNavItems.push({ ...sym, groupTitle: group.title, moduleTitle: item.title });
            if (sym.id === docId) {
              currentItem = sym;
              currentGroup = group;
              parentModule = item;
            }
          });
        }
      });
    });

    if (!currentItem) {
      currentItem = allNavItems[0];
      currentGroup = data.navigation[0];
      currentDocId = currentItem.id;
    }

    renderNavTree();
    updateActiveNav();

    // Breadcrumbs
    if (parentModule) {
      breadcrumbsEl.innerHTML = `
        <span>Docs</span>
        <span class="breadcrumb-separator">/</span>
        <a href="#${parentModule.id}">${parentModule.title}</a>
        <span class="breadcrumb-separator">/</span>
        <span class="breadcrumb-current">${currentItem.title}</span>
      `;
    } else {
      breadcrumbsEl.innerHTML = `
        <span>Docs</span>
        <span class="breadcrumb-separator">/</span>
        <span>${currentGroup.title}</span>
        <span class="breadcrumb-separator">/</span>
        <span class="breadcrumb-current">${currentItem.title}</span>
      `;
    }

    // Render Article Body
    const rawMarkdown = data.documents[currentDocId] || '# Document Not Found\nThe requested document does not exist.';
    docArticleEl.innerHTML = parseMarkdown(rawMarkdown);

    // Render LaTeX Math Formulas with KaTeX
    function applyKaTeX() {
      if (window.renderMathInElement) {
        try {
          renderMathInElement(docArticleEl, {
            delimiters: [
              { left: '$$', right: '$$', display: true },
              { left: '$', right: '$', display: false }
            ],
            throwOnError: false,
            ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
          });
        } catch (err) {
          console.warn('KaTeX rendering error:', err);
        }
      }
    }

    applyKaTeX();
    if (!window.renderMathInElement) {
      window.addEventListener('load', applyKaTeX, { once: true });
    }


    // Setup Copy Buttons
    document.querySelectorAll('.copy-code-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const code = decodeURIComponent(btn.getAttribute('data-code'));
        navigator.clipboard.writeText(code).then(() => {
          btn.classList.add('copied');
          btn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
            Copied!
          `;
          setTimeout(() => {
            btn.classList.remove('copied');
            btn.innerHTML = `
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
              Copy
            `;
          }, 2000);
        });
      });
    });

    // Intercept internal article link clicks to guarantee SPA navigation
    docArticleEl.querySelectorAll('a').forEach(a => {
      const href = a.getAttribute('href');
      if (href && (href.startsWith('#') || href.endsWith('.md'))) {
        a.addEventListener('click', (e) => {
          if (!href.startsWith('http://') && !href.startsWith('https://')) {
            e.preventDefault();
            const targetHash = resolveMarkdownLink(href);
            window.location.hash = targetHash;
          }
        });
      }
    });

    // Build Table of Contents
    buildTableOfContents();

    // Build Pagination
    buildPagination(allNavItems);

    // Scroll to anchor or top
    if (currentAnchorId) {
      setTimeout(() => {
        const el = document.getElementById(currentAnchorId);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth' });
        }
      }, 50);
    } else {
      window.scrollTo({ top: 0, behavior: 'instant' });
    }
  }

  function buildTableOfContents() {
    const headings = docArticleEl.querySelectorAll('h2, h3');
    if (headings.length === 0) {
      tocNavEl.innerHTML = '<span class="toc-link" style="color:var(--text-muted);">Overview</span>';
      return;
    }

    let html = '';
    headings.forEach(h => {
      const level = h.tagName.toLowerCase();
      const rawTitle = h.textContent.trim();
      let displayTitle = rawTitle;
      if (level === 'h3' && rawTitle.includes('.')) {
        const parts = rawTitle.split('.');
        displayTitle = '.' + parts[parts.length - 1];
      }
      const id = h.id;
      html += `<a href="#${currentDocId}#${id}" class="toc-link ${level === 'h3' ? 'toc-h3' : ''}" data-target-id="${id}" title="${escapeHtml(rawTitle)}">${escapeHtml(displayTitle)}</a>`;
    });
    tocNavEl.innerHTML = html;

    setupTocScrollSpy(headings);
  }


  function setupTocScrollSpy(headings) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          document.querySelectorAll('.toc-link').forEach(link => {
            if (link.getAttribute('data-target-id') === id) {
              link.classList.add('active');
            } else {
              link.classList.remove('active');
            }
          });
        }
      });
    }, { rootMargin: '0px 0px -70% 0px' });

    headings.forEach(h => observer.observe(h));
  }

  function buildPagination(allItems) {
    const currentIndex = allItems.findIndex(item => item.id === currentDocId);
    let prevHtml = '';
    let nextHtml = '';

    if (currentIndex > 0) {
      const prev = allItems[currentIndex - 1];
      prevHtml = `
        <a href="#${prev.id}" class="pagination-btn pagination-prev">
          <span class="pagination-label">← Previous</span>
          <span class="pagination-title">${prev.title}</span>
        </a>
      `;
    } else {
      prevHtml = '<div></div>';
    }

    if (currentIndex < allItems.length - 1) {
      const next = allItems[currentIndex + 1];
      nextHtml = `
        <a href="#${next.id}" class="pagination-btn pagination-next">
          <span class="pagination-label">Next →</span>
          <span class="pagination-title">${next.title}</span>
        </a>
      `;
    } else {
      nextHtml = '<div></div>';
    }

    docPaginationEl.innerHTML = prevHtml + nextHtml;
  }

  // ==========================================
  // 5. Search Modal & Index
  // ==========================================
  function openSearch() {
    searchModalEl.classList.add('open');
    searchModalEl.setAttribute('aria-hidden', 'false');
    searchInputEl.value = '';
    searchInputEl.focus();
    performSearch('');
  }

  function closeSearch() {
    searchModalEl.classList.remove('open');
    searchModalEl.setAttribute('aria-hidden', 'true');
  }

  function performSearch(query) {
    const q = query.trim().toLowerCase();
    let results = [];

    if (!q) {
      results = data.searchIndex.slice(0, 8);
    } else {
      results = data.searchIndex.filter(item => {
        return item.name.toLowerCase().includes(q) ||
               item.module.toLowerCase().includes(q) ||
               (item.summary && item.summary.toLowerCase().includes(q));
      }).slice(0, 10);
    }

    currentSearchResults = results;
    searchSelectedIdx = 0;
    renderSearchResults();
  }

  function renderSearchResults() {
    if (currentSearchResults.length === 0) {
      searchResultsEl.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-muted);">No symbols found matching your query.</div>';
      return;
    }

    let html = '';
    currentSearchResults.forEach((res, idx) => {
      const isSelected = idx === searchSelectedIdx;
      html += `
        <a href="#${res.ref_file}" class="search-result-item ${isSelected ? 'selected' : ''}" data-idx="${idx}">
          <div class="search-result-main">
            <span class="search-result-title">${escapeHtml(res.name)}</span>
            <span class="search-result-desc">${escapeHtml(res.summary || res.file)}</span>
          </div>
          <span class="search-result-badge">${res.module}</span>
        </a>
      `;
    });
    searchResultsEl.innerHTML = html;

    document.querySelectorAll('.search-result-item').forEach(item => {
      item.addEventListener('click', () => {
        closeSearch();
      });
    });
  }

  globalSearchBtn.addEventListener('click', openSearch);
  searchCloseBtn.addEventListener('click', closeSearch);

  searchInputEl.addEventListener('input', (e) => {
    performSearch(e.target.value);
  });

  window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      openSearch();
    } else if (e.key === '/' && document.activeElement !== searchInputEl && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
      e.preventDefault();
      openSearch();
    } else if (e.key === 'Escape' && searchModalEl.classList.contains('open')) {
      closeSearch();
    } else if (searchModalEl.classList.contains('open')) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        searchSelectedIdx = (searchSelectedIdx + 1) % currentSearchResults.length;
        renderSearchResults();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        searchSelectedIdx = (searchSelectedIdx - 1 + currentSearchResults.length) % currentSearchResults.length;
        renderSearchResults();
      } else if (e.key === 'Enter' && currentSearchResults[searchSelectedIdx]) {
        e.preventDefault();
        const target = currentSearchResults[searchSelectedIdx];
        window.location.hash = `#${target.ref_file}`;
        closeSearch();
      }
    }
  });

  searchModalEl.addEventListener('click', (e) => {
    if (e.target === searchModalEl) {
      closeSearch();
    }
  });

  mobileMenuBtn.addEventListener('click', () => {
    sidebarEl.classList.toggle('open');
  });

  navTreeEl.addEventListener('click', (e) => {
    if (e.target.classList.contains('nav-item-link') || e.target.classList.contains('nav-subitem-link')) {
      sidebarEl.classList.remove('open');
    }
  });

  // ==========================================
  // 6. Router (Hash-Based)
  // ==========================================
  function handleRoute() {
    let hash = window.location.hash.replace(/^#/, '');
    if (!hash) {
      hash = 'tutorials/quickstart';
    }

    const parts = hash.split('#');
    const docId = parts[0];
    const anchorId = parts[1] || '';

    renderDocument(docId, anchorId);
  }

  window.addEventListener('hashchange', handleRoute);

  // ==========================================
  // Initialization
  // ==========================================
  initTheme();
  handleRoute();

})();
