#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared HTML helpers for Shenbao interpretation dashboards."""

from __future__ import annotations

import re


CONTROLS_COLLAPSE_STYLE_ID = "codex-controls-collapse-style"
CONTROLS_COLLAPSE_SCRIPT_ID = "codex-controls-collapse-script"

CONTROLS_COLLAPSE_STYLE = f"""
<style id="{CONTROLS_COLLAPSE_STYLE_ID}">
  .codex-controls-toggle {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin: 8px 0 10px;
    padding: 6px 10px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    background: #fff;
    color: #334155;
    font: inherit;
    font-size: 12px;
    line-height: 1.2;
    cursor: pointer;
  }}
  .codex-controls-toggle:hover {{
    border-color: #94a3b8;
    background: #f8fafc;
  }}
  .codex-controls-toggle[aria-expanded="false"] {{
    margin-bottom: 0;
  }}
  .codex-controls-hidden {{
    display: none !important;
  }}
</style>
"""

CONTROLS_COLLAPSE_SCRIPT = f"""
<script id="{CONTROLS_COLLAPSE_SCRIPT_ID}">
(function() {{
  function makeButton(targets) {{
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'codex-controls-toggle';
    button.setAttribute('aria-expanded', 'true');
    button.textContent = 'Hide controls';
    button.addEventListener('click', function() {{
      var expanded = button.getAttribute('aria-expanded') === 'true';
      targets.forEach(function(el) {{ el.classList.toggle('codex-controls-hidden', expanded); }});
      button.setAttribute('aria-expanded', expanded ? 'false' : 'true');
      button.textContent = expanded ? 'Show controls' : 'Hide controls';
    }});
    return button;
  }}
  function isHeaderControl(el) {{
    if (!el || !el.matches) return false;
    if (el.matches('h1,h2,h3,.note,.codex-controls-toggle')) return false;
    if (el.matches('.controls,.checkbox-line,.region-toolbar,.region-list,.pos-toolbar,.pos-list,.check-panel')) return true;
    return !!el.querySelector('select,input,button');
  }}
  function install() {{
    document.querySelectorAll('header').forEach(function(header) {{
      var controls = header.querySelector('.controls');
      if (!controls || header.dataset.codexControlsCollapseReady === '1') return;
      var children = Array.prototype.slice.call(header.children);
      var targets = children.filter(isHeaderControl);
      if (!targets.length) return;
      targets.forEach(function(el) {{ el.dataset.codexControlsGrouped = '1'; }});
      var button = makeButton(targets);
      header.insertBefore(button, targets[0]);
      header.dataset.codexControlsCollapseReady = '1';
    }});
    document.querySelectorAll('.controls').forEach(function(panel) {{
      if (panel.dataset.codexControlsGrouped === '1' || panel.dataset.codexControlsCollapseReady === '1') return;
      if (!panel.parentNode) return;
      var button = makeButton([panel]);
      panel.parentNode.insertBefore(button, panel);
      panel.dataset.codexControlsCollapseReady = '1';
    }});
  }}
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', install);
  }} else {{
    install();
  }}
}})();
</script>
"""


def inject_controls_collapse(html: str) -> str:
    """Inject collapsible dashboard-control behavior into generated HTML."""
    if 'class="controls"' not in html and "class='controls'" not in html:
        return html

    style_re = re.compile(
        r'\n?<style id="' + re.escape(CONTROLS_COLLAPSE_STYLE_ID) + r'">.*?</style>\n?',
        re.DOTALL,
    )
    script_re = re.compile(
        r'\n?<script id="' + re.escape(CONTROLS_COLLAPSE_SCRIPT_ID) + r'">.*?</script>\n?',
        re.DOTALL,
    )
    out = style_re.sub("", html)
    out = script_re.sub("", out)
    if "</head>" in out:
        out = out.replace("</head>", CONTROLS_COLLAPSE_STYLE + "</head>", 1)
    if "</body>" in out:
        out = out.replace("</body>", CONTROLS_COLLAPSE_SCRIPT + "</body>", 1)
    elif "</html>" in out:
        out = out.replace("</html>", CONTROLS_COLLAPSE_SCRIPT + "</html>", 1)
    return out
