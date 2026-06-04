/* ════════════════════════════════════════════════════════════════════
   CoS // AUTH GATE
   Пароль на весь сайт без cookie: токен живёт ТОЛЬКО в памяти вкладки и
   шлётся в заголовке Authorization: Bearer. При обновлении страницы (F5)
   память обнуляется → токена нет → оверлей снова просит пароль.

   Перехватывает window.fetch ЛЕНИВО: заголовок добавляется только когда
   токен уже есть, а оверлей всплывает лишь при ответе 401. Поэтому если
   на сервере SITE_PASSWORD не задан (локальная разработка) — 401 не будет
   и оверлей никогда не покажется.
   ════════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  var token = null;       // подписанный токен (в памяти вкладки)
  var pending = null;     // промис, пока ждём ввод пароля (дедуп параллельных 401)

  var origFetch = window.fetch.bind(window);

  function isCrossOrigin(url) {
    return /^https?:\/\//i.test(url) && url.indexOf(location.origin) !== 0;
  }

  function withAuth(init) {
    init = init || {};
    var headers = new Headers(init.headers || {});
    headers.set("Authorization", "Bearer " + token);
    init.headers = headers;
    return init;
  }

  window.fetch = async function (input, init) {
    var url = typeof input === "string" ? input : (input && input.url) || "";
    var skip = isCrossOrigin(url) || url.indexOf("/login") !== -1;

    if (!skip && token) init = withAuth(init);

    var res = await origFetch(input, init);

    if (res.status === 401 && !skip) {
      // Нет токена или протух — просим пароль и повторяем запрос.
      // pending НЕ трогаем: им владеет ensureAuth/showGate (дедуп параллельных
      // 401, иначе два запроса показали бы два оверлёя).
      token = null;
      await ensureAuth();
      res = await origFetch(input, withAuth(init));
    }
    return res;
  };

  function ensureAuth() {
    if (token) return Promise.resolve();
    if (!pending) pending = showGate();
    return pending;
  }

  // ── Оверлей входа ──────────────────────────────────────────────────
  function injectStyleOnce() {
    if (document.getElementById("agp-style")) return;
    var st = document.createElement("style");
    st.id = "agp-style";
    st.textContent = [
      "@keyframes agp-runline{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}",
      "@keyframes agp-flicker{0%,96%,100%{opacity:1}97%{opacity:.55}98%{opacity:1}99%{opacity:.7}}",
      "#agp-ov{position:fixed;inset:0;z-index:2147483647;display:flex;align-items:center;justify-content:center;",
      "  font-family:'Space Grotesk','Inter',ui-sans-serif,sans-serif;color:#e8e9ff;",
      "  background:radial-gradient(1200px 800px at 20% 10%,rgba(107,92,255,.10),transparent 60%),",
      "  radial-gradient(900px 700px at 90% 80%,rgba(0,240,255,.06),transparent 65%),#050614;}",
      "#agp-ov::after{content:'';position:absolute;inset:0;pointer-events:none;mix-blend-mode:overlay;",
      "  background:repeating-linear-gradient(to bottom,transparent 0,transparent 2px,rgba(0,0,0,.18) 2px,rgba(0,0,0,.18) 3px);}",
      "#agp-ov *{box-sizing:border-box;margin:0;padding:0;}",
      ".agp-panel{--c:14px;position:relative;padding:1px;width:100%;max-width:380px;",
      "  background:linear-gradient(135deg,#2a2d7a,#1f2160 40%,#2a2d7a);",
      "  clip-path:polygon(var(--c) 0,100% 0,100% calc(100% - var(--c)),calc(100% - var(--c)) 100%,0 100%,0 var(--c));}",
      ".agp-inner{position:relative;padding:34px 30px 30px;",
      "  background:linear-gradient(180deg,rgba(15,16,48,.94),rgba(10,10,31,.94));",
      "  clip-path:polygon(var(--c) 0,100% 0,100% calc(100% - var(--c)),calc(100% - var(--c)) 100%,0 100%,0 var(--c));}",
      ".agp-runline{position:absolute;left:0;right:0;top:0;height:1px;overflow:hidden;opacity:.6;",
      "  background:linear-gradient(90deg,transparent,#00f0ff,transparent);}",
      ".agp-runline::after{content:'';position:absolute;inset:0;animation:agp-runline 3.6s linear infinite;",
      "  background:linear-gradient(90deg,transparent 0%,#00f0ff 50%,transparent 100%);}",
      ".agp-brand{display:flex;align-items:baseline;gap:10px;margin-bottom:4px;}",
      ".agp-title{font-weight:700;font-size:26px;letter-spacing:.04em;position:relative;display:inline-block;animation:agp-flicker 5.4s infinite;}",
      ".agp-title::before,.agp-title::after{content:attr(data-text);position:absolute;left:0;top:0;width:100%;pointer-events:none;mix-blend-mode:screen;}",
      ".agp-title::before{color:#ff2e63;transform:translate(1.2px,0);clip-path:inset(0 0 60% 0);opacity:.85;}",
      ".agp-title::after{color:#00f0ff;transform:translate(-1.2px,0);clip-path:inset(58% 0 0 0);opacity:.85;}",
      ".agp-sub{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:10px;color:#6e74a8;letter-spacing:.06em;}",
      ".agp-hud{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:10px;color:#6e74a8;letter-spacing:.14em;text-transform:uppercase;margin:18px 0 8px;}",
      ".agp-input{width:100%;padding:12px 14px;font-family:'JetBrains Mono',ui-monospace,monospace;font-size:13px;letter-spacing:.04em;",
      "  background:rgba(5,6,20,.8);color:#e8e9ff;border:1px solid #2a2d7a;outline:0;transition:border-color .15s,box-shadow .15s;",
      "  clip-path:polygon(8px 0,100% 0,100% calc(100% - 8px),calc(100% - 8px) 100%,0 100%,0 8px);}",
      ".agp-input::placeholder{color:#3e4380;}",
      ".agp-input:focus{border-color:#00f0ff;box-shadow:0 0 20px rgba(0,240,255,.18);}",
      ".agp-btn{width:100%;margin-top:14px;padding:11px 16px;font-family:'JetBrains Mono',ui-monospace,monospace;font-size:11px;font-weight:600;",
      "  letter-spacing:.12em;text-transform:uppercase;background:#00f0ff;color:#021016;border:1px solid #00f0ff;cursor:pointer;",
      "  clip-path:polygon(8px 0,100% 0,100% calc(100% - 8px),calc(100% - 8px) 100%,0 100%,0 8px);box-shadow:0 0 18px rgba(0,240,255,.35);transition:background .15s,box-shadow .15s;}",
      ".agp-btn:hover{background:#5cf8ff;box-shadow:0 0 30px rgba(0,240,255,.6);}",
      ".agp-btn:disabled{opacity:.55;cursor:default;box-shadow:none;}",
      ".agp-err{margin-top:12px;min-height:16px;font-family:'JetBrains Mono',ui-monospace,monospace;font-size:11px;letter-spacing:.04em;color:#ff2e63;text-shadow:0 0 8px rgba(255,46,99,.5);}"
    ].join("");
    document.head.appendChild(st);
  }

  function showGate() {
    return new Promise(function (resolve) {
      injectStyleOnce();
      var ov = document.createElement("div");
      ov.id = "agp-ov";
      ov.innerHTML =
        '<form class="agp-panel" id="agp-form">' +
        '  <div class="agp-inner">' +
        '    <div class="agp-runline"></div>' +
        '    <div class="agp-brand">' +
        '      <span class="agp-title" data-text="ШТАБ">ШТАБ</span>' +
        '      <span class="agp-sub">// CHIEF.OF.STAFF</span>' +
        '    </div>' +
        '    <div class="agp-hud">[ ДОСТУП ОГРАНИЧЕН // ENTER PASSCODE ]</div>' +
        '    <input class="agp-input" type="password" id="agp-pwd" placeholder="• • • • • • • •" autocomplete="current-password">' +
        '    <button class="agp-btn" type="submit" id="agp-btn">Войти →</button>' +
        '    <div class="agp-err" id="agp-err"></div>' +
        '  </div>' +
        '</form>';

      function mount() {
        document.body.appendChild(ov);
        var form = ov.querySelector("#agp-form");
        var pwd = ov.querySelector("#agp-pwd");
        var err = ov.querySelector("#agp-err");
        var btn = ov.querySelector("#agp-btn");
        pwd.focus();
        form.addEventListener("submit", async function (e) {
          e.preventDefault();
          err.textContent = "";
          btn.disabled = true;
          try {
            var r = await origFetch("/login", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ password: pwd.value })
            });
            if (r.ok) {
              var data = await r.json();
              token = data.token;
              ov.remove();
              pending = null;
              resolve();
            } else {
              err.textContent = "> ОТКАЗ: неверный пароль";
              pwd.select();
            }
          } catch (_) {
            err.textContent = "> ОШИБКА СЕТИ — повторите";
          } finally {
            btn.disabled = false;
          }
        });
      }

      if (document.body) mount();
      else document.addEventListener("DOMContentLoaded", mount);
    });
  }
})();
