// usage: node scrape.js outdir url1 name1 url2 name2 ...
const { chromium } = require('/opt/node-tools/node_modules/playwright');
const fs = require('fs');
(async () => {
  const [outdir, ...rest] = process.argv.slice(2);
  fs.mkdirSync(outdir, {recursive: true});
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox','--ignore-certificate-errors'] });
  const ctx = await browser.newContext({ userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36', locale: 'en-US' });
  for (let i = 0; i < rest.length; i += 2) {
    const url = rest[i], name = rest[i+1];
    const page = await ctx.newPage();
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
      await page.waitForTimeout(6000);
      const txt = await page.evaluate(() => document.body ? document.body.innerText : '');
      fs.writeFileSync(`${outdir}/${name}.txt`, txt);
      console.log(name, txt.length, (await page.title()).slice(0,80));
    } catch (e) { console.log(name, 'ERR', e.message.slice(0,120)); }
    await page.close();
  }
  await browser.close();
})();
