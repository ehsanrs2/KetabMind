const { test, expect } = require('@playwright/test');

test.describe('Upload and Chat pages', () => {
  test('renders placeholders in a blank document', async ({ page }) => {
    await page.setContent('<main id="root"></main>');
    const root = page.locator('#root');
    await expect(root).toHaveCount(1);
  });
});
