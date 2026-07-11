import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  
  console.log("Navigating to https://enginewatch.tech");
  await page.goto('https://enginewatch.tech', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000); 
  
  // Try to find the new EngineSelector or AnomalyScatter
  const selectors = [
    'text=/Engine 34/', 
    'text=/FD001/',
    '.recharts-scatter-symbol', // AnomalyScatter dot
    '.recharts-layer', 
    'a[href^="/engine/"]'
  ];
  
  let clicked = false;
  for (const sel of selectors) {
      try {
          const loc = page.locator(sel).first();
          if (await loc.count() > 0) {
              console.log("Clicking selector: " + sel);
              // Force click in case it's an SVG element or intercepted
              await loc.click({ force: true });
              clicked = true;
              break;
          }
      } catch (e) {}
  }
  
  if (clicked) {
      try {
          await page.waitForURL('**/engine/**', { timeout: 10000 });
          const url = new URL(page.url());
          if (!url.pathname.startsWith('/engine/')) {
              console.error("URL pathname did not change to /engine/:id route. It is: " + url.pathname);
              process.exit(1);
          }
          console.log("Successfully navigated to: " + url.pathname);
          
          await page.waitForTimeout(3000); 
          await page.screenshot({ path: 'assets/screenshot-engine-drilldown.png', fullPage: true });
          console.log("Saved engine drilldown screenshot");
      } catch (e) {
          console.error("Failed to navigate to engine route:", e.message);
          process.exit(1);
      }
  } else {
      console.log("Failed to click an engine component.");
      process.exit(1);
  }
  
  await browser.close();
})();
