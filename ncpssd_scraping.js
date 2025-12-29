(async () => {
  "use strict";
  /************** 실행 시 사용자 입력 **************/
  const askParams = () => {
    // 페이지 루프 횟수 입력 (기본 100)
    let maxPagesInput = prompt("최대 순회 페이지 수를 입력하세요 (예: 100)", "100");
    let maxPages = parseInt((maxPagesInput || "").trim(), 10);
    if (!Number.isFinite(maxPages) || maxPages <= 0) {
      alert("유효하지 않은 값입니다. 기본값 100을 사용합니다.");
      maxPages = 100;
    }
    // 파일명 슬러그 입력 (기본 'list')
    let slug = (prompt("저장 파일명 슬러그를 입력하세요 (예: list)", "list") || "")
      .trim()
      .replace(/[^\p{L}\p{N}_-]+/gu, ""); // 글자/숫자/언더스코어/하이픈만
    if (!slug) {
      alert("슬러그가 비어 있어 기본값 'list'를 사용합니다.");
      slug = "list";
    }
    return { maxPages, slug };
  };
  const { maxPages, slug } = askParams();

  /************** 공통 유틸 **************/
  const $all = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const $one = (sel, root = document) => root.querySelector(sel);
  const text = (node) => (node ? node.textContent.trim() : "").replace(/\s+/g, " ");
  const nowDate = (() => {
    const d = new Date();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    return `${d.getFullYear()}-${mm}-${dd}`;
  })();
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // XML escape
  const x = (s) =>
    String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&apos;");

  // 중국 숫자 → 아라비아
  const cn2num = (raw) => {
    if (!raw) return "";
    const s = String(raw).trim();
    if (/\d+/.test(s)) return s.match(/\d+/)[0];
    const map = { 零:0, 一:1, 二:2, 两:2, 三:3, 四:4, 五:5, 六:6, 七:7, 八:8, 九:9, 十:10 };
    if (s === "十") return "10";
    if (s.includes("十")) {
      const [a, b] = s.split("十");
      const tens = a ? map[a] : 1;
      const ones = b ? map[b] : 0;
      return String(tens * 10 + ones);
    }
    return String(map[s] ?? s);
  };

  const findLineByLabel = (li, labels) => {
    const nodes = $all("p, div, span, li", li);
    for (const n of nodes) {
      const t = text(n);
      if (!t) continue;
      for (const lb of labels) if (t.includes(lb)) return t;
    }
    return "";
  };

  const parseSource = (line) => {
    const journal = (line?.match(/《([^》]+)》/) || [])[1] || "";
    const year = (line?.match(/(\d{4})年/) || [])[1] || "";
    const issueRaw = (line?.match(/第\s*([0-9一二三四五六七八九十两]+)\s*期/) || [])[1] || "";
    const issue = cn2num(issueRaw);
    const pageStart = (line?.match(/(\d+)\s*-\s*(\d+)/) || [])[1] || "";
    const pageEnd = (line?.match(/(\d+)\s*-\s*(\d+)/) || [])[2] || "";
    return { journal, year, issue, pageStart, pageEnd };
  };

  const parseFunding = (line) => {
    const acknowledgement = (line || "").replace(/^\s*基金项目[:：]\s*/, "").trim();
    const researchtitle = (line?.match(/“([^”]+)”/) || [])[1] || "";
    const number = (line?.match(/[（(]([A-Za-z0-9\-]+)[)）]/) || [])[1] || "";
    let order = "";
    const m1 = line?.match(/本文系(.+?项目)/) || line?.match(/系(.+?项目)/);
    if (m1 && m1[1]) order = m1[1].trim();
    return { acknowledgement, order, researchtitle, number };
  };

  /******** 라벨 정규화 & 라벨기반 파싱 유틸 ********/
  const NORM = (s) => String(s ?? "")
    .replace(/[\s\u3000]/g, "")      // 모든 공백 제거
    .replace(/[：:]/g, "")           // 콜론 제거
    .replace(/關鍵詞|关键字/g, "关键词")
    .replace(/摘要/g, "摘要")
    .replace(/作{1}者/g, "作者");

  // <p>에서 "라벨: 값" 구조를 안정적으로 분리
  const splitLabelValueFromP = (p) => {
    const nodes = Array.from(p.childNodes);
    let labelRaw = "";
    let valueParts = [];
    let inValue = false;

    for (const node of nodes) {
      const t = (node.textContent || "").replace(/\s+/g, " ").trim();
      if (!t) continue;

      if (!inValue) {
        labelRaw += t;
        if (/[：:]/.test(t)) {
          const after = t.split(/[:：]/).slice(-1)[0].trim();
          if (after) valueParts.push(after);
          inValue = true;
        }
      } else {
        valueParts.push(t);
      }
    }

    const labelNorm = NORM(labelRaw);
    const valueText = valueParts.join(" ").replace(/\s+/g, " ").trim();
    return { labelNorm, valueText };
  };

  // 라벨 후보 배열로 해당 <p>의 값만 추출
  const extractValueByLabel = (li, targets) => {
    const targetSet = new Set(targets.map(NORM));
    for (const p of $all("p", li)) {
      const { labelNorm, valueText } = splitLabelValueFromP(p);
      if (targetSet.has(labelNorm)) return valueText;
    }
    return "";
  };

  /********** 정확한 제목/저자/키워드/초록 추출 (라벨 기반으로 교체) **********/
  // 제목: li 내 첫 번째 제목 링크만
  const parseTitleStrict = (li) => {
    const titleEl =
      $one("div > a:first-child", li) ||
      $one("h1 > a:first-child, h2 > a:first-child, h3 > a:first-child, .title a:first-child", li);
    return text(titleEl);
  };

  // 저자: "作　　者" 등 변형 포함 → "作者"로 정규화 매칭
  const parseAuthorsStrict = (li) => {
    const raw = extractValueByLabel(li, ["作者", "作 者", "作　　者"]);
    if (!raw) return [];
    return raw
      .split(/[、，,；;·\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
  };

  // 키워드: "关 键 词" 변형 포함 → "关键词"로 정규화 매칭
  const parseKeywordsStrict = (li) => {
    const raw = extractValueByLabel(li, ["关键词", "关 键 词", "關鍵詞", "关键字"]);
    if (!raw) return [];
    return raw
      .split(/[、，,；;|｜\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
  };

  // 초록: "摘　　要" 변형 포함 → "摘要"로 정규화 매칭
  const parseAbstractStrict = (li) => {
    return extractValueByLabel(li, ["摘要", "摘 要", "摘　　要"]) || "";
  };

  const parseImpact = (li) => {
    const t = text(li);
    const download =
      (t.match(/下载次数[:：]\s*(\d+)/) || [])[1] ||
      (t.match(/下载量[:：]\s*(\d+)/) || [])[1] || "0";
    const read =
      (t.match(/在线阅读[:：]\s*(\d+)/) || [])[1] ||
      (t.match(/阅读量[:：]\s*(\d+)/) || [])[1] || "0";
    return { download, read };
  };

  /**************** 페이지 파싱 함수 (현재 페이지) ****************/
  const parseCurrentPage = () => {
    const items = $all("#ul_articlelist > li");
    const results = [];

    for (const li of items) {
      const title = parseTitleStrict(li);

      const sourceLine = findLineByLabel(li, ["出处", "出 处"]);
      const { journal, year, issue, pageStart, pageEnd } = parseSource(sourceLine);

      // === 라벨 기반 파싱으로 교체 ===
      const authors = parseAuthorsStrict(li);
      const keywords = parseKeywordsStrict(li);

      const absLine = findLineByLabel(li, ["摘要", "摘 要"]);
      const abs = absLine ? absLine.replace(/^\s*摘\s*要[:：]?\s*/, "").trim() : "";
      const fundingLine = findLineByLabel(li, ["基金项目", "基金項目"]) || "";
      const funding = fundingLine
        ? parseFunding(fundingLine)
        : { acknowledgement: "", order: "", researchtitle: "", number: "" };

      const impact = parseImpact(li);

      // 동일성 체크용 키(중복 방지): 제목+저널+연도+시작쪽
      const key = [title, journal, year, pageStart].join("||");

      // XML 조각 (구조 유지)
      let xml = "  <article>\n";
      xml += `    <title>${x(title)}</title>\n`;
      xml += `    <journal>${x(journal)}</journal>\n`;
      xml += `    <year>${x(year)}</year>\n`;
      xml += `    <issue>${x(issue)}</issue>\n`;
      xml += "    <page>\n";
      xml += `      <startpage>${x(pageStart)}</startpage>\n`;
      xml += `      <endpage>${x(pageEnd)}</endpage>\n`;
      xml += "    </page>\n";

      xml += "    <authors>\n";
      for (const a of authors) xml += `      <author>${x(a)}</author>\n`;
      xml += "    </authors>\n";

      xml += "    <funding>\n";
      xml += `      <acknowledgement>${x(funding.acknowledgement)}</acknowledgement>\n`;
      xml += `      <order>${x(funding.order)}</order>\n`;
      xml += `      <researchtitle>${x(funding.researchtitle)}</researchtitle>\n`;
      xml += `      <number>${x(funding.number)}</number>\n`;
      xml += "    </funding>\n";

      xml += `    <abstract>${x(abs)}</abstract>\n`;

      xml += "    <keywords>\n";
      for (const k of keywords) xml += `      <keyword>${x(k)}</keyword>\n`;
      xml += "    </keywords>\n";

      xml += "    <impact>\n";
      xml += `      <date>${x(nowDate)}</date>\n`;
      xml += `      <download>${x(impact.download)}</download>\n`;
      xml += `      <read>${x(impact.read)}</read>\n`;
      xml += "    </impact>\n";

      xml += "  </article>";

      results.push({ key, xml });
    }
    return results;
  };

  /**************** 페이지 전환 대기 (다음 페이지 로딩 감지) ****************/
  const waitForListChange = (prevSignature, timeoutMs = 10000) =>
    new Promise((resolve, reject) => {
      const list = $one("#ul_articlelist");
      if (!list) return resolve(true); // 없으면 일단 통과

      const timer = setTimeout(() => {
        obs.disconnect();
        resolve(false); // 시간 초과
      }, timeoutMs);

      const obs = new MutationObserver(() => {
        const sig = text($one("#ul_articlelist > li:first-child"));
        if (sig && sig !== prevSignature) {
          clearTimeout(timer);
          obs.disconnect();
          resolve(true);
        }
      });
      obs.observe(list, { childList: true, subtree: true, characterData: true });
    });

  /******** 페이지네이션 루프 ********/
  const allXML = [];
  const seen = new Set(); // 중복 방지
  let pageCount = 0;
  while (pageCount < maxPages) {
    pageCount++;
    console.log(`▶ 페이지 ${pageCount} 파싱 중…`);
    const prevSig = text($one("#ul_articlelist > li:first-child")) || Math.random().toString(36);

    // 현재 페이지 파싱
    const items = parseCurrentPage();
    for (const it of items) {
      if (it.key && !seen.has(it.key)) {
        seen.add(it.key);
        allXML.push(it.xml);
      }
    }
    console.log(`… 누적 기사 수: ${allXML.length}`);

    // 다음 버튼 확인
    const nextBtn = $one('#layui-laypage-0 a.layui-laypage-next');
    const nextBtnText = text(nextBtn);
    const disabled = !nextBtn || nextBtn.classList.contains("layui-disabled") || !/下一页/.test(nextBtnText || "");

    if (disabled) {
      console.log("◼ 마지막 페이지로 판단되어 수집 종료.");
      break;
    }
    if (pageCount >= maxPages) {
      console.warn(`◼ maxPages(${maxPages})에 도달하여 중단합니다.`);
      break;
    }

    // 다음 페이지 클릭
    nextBtn.click();

    // 로딩 대기: MutationObserver로 목록 변화 감지 + 약간의 여유
    const changed = await waitForListChange(prevSig, 2000);
    if (!changed) {
      // 변화가 없으면 잠깐 더 대기 후 한 번 더 점검
      await sleep(1200);
      const sigNow = text($one("#ul_articlelist > li:first-child"));
      if (sigNow === prevSig) {
        console.warn("목록이 갱신되지 않았습니다. 수집을 종료합니다.");
        break;
      }
    }
    // 서버 부하/차단 회피용 짧은 휴식
    await sleep(300);
  }

  /********************** 단일 XML 파일로 저장 **********************/
  const xmlDoc = `<?xml version="1.0" encoding="utf-8"?>\n<articles>\n${allXML.join("\n")}\n</articles>\n`;
  const blob = new Blob([xmlDoc], { type: "text/xml;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `articles_${slug}_${nowDate}.xml`;
  document.body.appendChild(a);
  a.click();
  URL.revokeObjectURL(a.href);
  a.remove();
  console.log(`✅ 완료: 총 ${allXML.length}편을 수집하여 단일 XML로 저장했습니다. (파일명: articles_${slug}_${nowDate}.xml, ${pageCount}페이지 처리)`);
})();
