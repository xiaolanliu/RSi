// Optional browser check. npm install --no-save playwright && npx playwright install chromium
const fs = require('fs');
const {chromium} = require(process.env.RSI_PLAYWRIGHT_MODULE || 'playwright');
let browser;
(async () => {
  const base = process.env.RSI_REPORT_BASE_URL || 'http://127.0.0.1:8000';
  browser = await chromium.launch({headless:true,
    ...(process.env.RSI_CHROMIUM ? {executablePath:process.env.RSI_CHROMIUM} : {})});
  const page = await browser.newPage({viewport:{width:1280,height:950}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const results = [];
  for (const name of ['pant_fail_ep0','offline_2000']) {
    const url = `${base}/reports/${name}/`;
    const response = await page.goto(url);
    if (!response.ok()) throw new Error(`Page failed: ${name}`);
    await page.waitForFunction(() => Number(document.getElementById('frame').max) > 0);
    await page.locator('#frame').evaluate((node, value) => {
      node.value=value; node.dispatchEvent(new Event('input',{bubbles:true}));
    }, name==='pant_fail_ep0' ? '303' : '238');
    const current = await page.locator('#current').textContent();
    if (name==='pant_fail_ep0' && !current.includes('报警 是')) throw new Error('Expected frame 303 alarm');
    for (const file of ['predictions.csv','predictions.npz','summary.json']) {
      const result = await page.request.get(url+file);
      if (!result.ok()) throw new Error(`Missing ${name}/${file}`);
    }
    await page.setViewportSize({width:390,height:844});
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Mobile overflow');
    await page.setViewportSize({width:1280,height:950});
    results.push({report:name, slider:true, downloads:true, mobile:true, passed:true});
  }
  if (errors.length) throw new Error(errors.join('\n'));
  fs.mkdirSync('validation',{recursive:true});
  fs.writeFileSync('validation/browser.json',JSON.stringify({passed:true,reports:results,javascript_errors:errors},null,2)+'\n');
  await page.goto(`${base}/reports/pant_fail_ep0/`);
  await page.locator('#frame').evaluate(node=>{node.value='303';node.dispatchEvent(new Event('input',{bubbles:true}));});
  await page.screenshot({path:'validation/replay_preview.png',fullPage:true});
  console.log(JSON.stringify(results));
  await browser.close();
})().catch(async error=>{console.error(error);if(browser)await browser.close();process.exitCode=1;});
