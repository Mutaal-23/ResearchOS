/*
 * Ask a question, render the answer and its citations.
 *
 * Deliberately no framework. This is ~120 lines of fetch and DOM work, and a
 * framework would add a build step and a dependency to render three elements.
 * The server returns HTML fragments for citations but the answer itself is
 * built here, so the same code path works whether or not the server found
 * evidence.
 */

const $ = (id) => document.getElementById(id);

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

/** Wrap [1] markers as citation pills pointing at the source list. */
function linkCitations(html) {
  return html.replace(/\[(\d+)\]/g, (match, number) => {
    const index = Number(number);
    return `<button class="cite" data-cite="${index}" title="Go to source ${index}">${index}</button>`;
  });
}

function renderAnswer(data) {
  $('no-evidence').classList.add('hidden');

  const body = linkCitations(escapeHtml(data.answer).replace(/\n{2,}/g, '</p><p>'));
  $('answer-body').innerHTML = `<p>${body}</p>`;
  $('answer-latency').textContent = `${Math.round(data.latency_ms)} ms`;

  // Say so when the ranking behind the citations is degraded. Without this the
  // source scores are raw rank-fusion values that look like relevance, and a
  // user has no way to know the reranker never ran.
  const warning = $('rerank-warning');
  if (data.reranked === false) {
    warning.textContent =
      data.rerank_note ||
      'Ranking is degraded: passages are in raw hybrid-search order.';
    warning.classList.remove('hidden');
  } else {
    warning.classList.add('hidden');
  }

  const sources = $('citations');
  if (!data.citations.length) {
    sources.innerHTML =
      '<p class="text-sm text-stone-500">No citations. This answer is not grounded in the corpus.</p>';
  } else {
    const label = data.reranked === false ? 'search rank' : 'score';
    sources.innerHTML = `
      <h3 class="text-sm font-semibold uppercase tracking-wide text-stone-500">Sources</h3>
      ${data.citations
        .map(
          (c) => `
        <div class="source" id="source-${c.index}">
          <div class="source-title">[${c.index}] ${
            c.section_title ? escapeHtml(c.section_title) : 'Untitled section'
          }</div>
          <div class="source-meta">${
            c.page_number ? `page ${c.page_number} · ` : ''
          }${label} ${c.score}</div>
          <div class="source-excerpt">${escapeHtml(c.excerpt)}</div>
        </div>`,
        )
        .join('')}
    `;
  }

  $('answer').classList.remove('hidden');
}

function renderNoEvidence(message) {
  $('answer').classList.add('hidden');
  $('no-evidence-body').textContent = message;
  $('no-evidence').classList.remove('hidden');
}

async function ask(question) {
  const button = $('ask-button');
  button.disabled = true;
  $('status').classList.remove('hidden');
  $('status').textContent = 'Searching the corpus…';

  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    $('status').classList.add('hidden');

    if (response.status === 429) {
      const detail = await response.json();
      renderNoEvidence(detail.detail || 'The embedding quota is exhausted. Try again later.');
      return;
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      renderNoEvidence(detail.detail || `Request failed (${response.status})`);
      return;
    }

    const data = await response.json();
    if (data.grounded) {
      renderAnswer(data);
    } else {
      renderNoEvidence(data.answer);
    }
  } catch (error) {
    $('status').classList.add('hidden');
    renderNoEvidence('Could not reach the server. Is it still running?');
    console.error(error);
  } finally {
    button.disabled = false;
  }
}

async function loadDocuments() {
  const container = $('documents');
  try {
    const response = await fetch('/api/documents');
    const data = await response.json();

    if (!data.total) {
      container.innerHTML = `
        <p class="text-sm text-stone-500">Nothing indexed yet.</p>
        <p class="source-meta mt-1">Run <code>researchos ingest FILE</code> to add a document.</p>
      `;
      return;
    }

    container.innerHTML = data.documents
      .map(
        (d) => `
      <div class="border-b border-stone-200 pb-2 text-sm last:border-0 dark:border-stone-800">
        <div class="font-medium truncate">${escapeHtml(d.title)}</div>
        <div class="source-meta">${d.n_chunks} chunks · ${d.n_pages ?? 0} pages · ${d.status}</div>
      </div>`,
      )
      .join('');
  } catch {
    container.innerHTML = '<p class="text-sm text-stone-500">Could not load documents.</p>';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  $('ask-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const question = $('question').value.trim();
    if (question) ask(question);
  });

  // Delegated so citation pills work for answers rendered after load.
  document.addEventListener('click', (event) => {
    const pill = event.target.closest('.cite');
    if (pill) {
      const target = $(`source-${pill.dataset.cite}`);
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        target.style.transition = 'background 240ms';
        target.style.background = 'color-mix(in srgb, var(--accent) 10%, transparent)';
        setTimeout(() => {
          target.style.background = 'transparent';
        }, 700);
      }
      return;
    }

    if (event.target.id === 'theme-toggle') {
      const root = document.documentElement;
      root.classList.toggle('dark');
      localStorage.theme = root.classList.contains('dark') ? 'dark' : 'light';
    }
  });

  loadDocuments();
});