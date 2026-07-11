import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  
  console.log("Navigating to https://enginewatch.tech");
  await page.goto('https://enginewatch.tech', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000); 
  
  await page.screenshot({ path: 'assets/screenshot-fleet-command.png', fullPage: true });
  console.log("Saved fleet command screenshot");
  
  const engineLocators = [
    'a[href*="/engine/"]', 
    'text=/Engine \\d+/',
    'text=/FD001/',
    '.engine-card', 
    'a'
  ];
  
  let clicked = false;
  for (const selector of engineLocators) {
      try {
          const loc = page.locator(selector).first();
          if (await loc.count() > 0) {
              console.log("Clicking selector: " + selector);
              await loc.click();
              clicked = true;
              break;
          }
      } catch (e) {}
  }
  
  if (!clicked) {
      console.log("Failed to click an engine. Please verify the DOM.");
  } else {
      await page.waitForTimeout(3000); 
      await page.screenshot({ path: 'assets/screenshot-engine-drilldown.png', fullPage: true });
      console.log("Saved engine drilldown screenshot");
  }
  
  await browser.close();
})();
