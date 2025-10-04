import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import path from 'path';

const sampleDocument = path.resolve(__dirname, '../../../docs/fixtures/sample.pdf');

async function performLogin(page: Page, email: string, password: string) {
  await page.goto('/login');
  await expect(page.getByRole('heading', { name: 'Login' })).toBeVisible();
  await page.getByLabel('Email').fill(email);
  await page.getByLabel('Password').fill(password);
  await Promise.all([
    page.waitForURL('**/chat'),
    page.getByRole('button', { name: /sign in/i }).click(),
  ]);
  await expect(page.getByRole('heading', { name: 'Sessions' })).toBeVisible();
}

test.describe('KetabMind end-to-end flow', () => {
  test('user can upload, index, chat with citations, and enforce isolation', async ({ page, browser }) => {
    const baseURL = test.info().project.use.baseURL ?? 'http://127.0.0.1:3100';

    await performLogin(page, 'alice@example.com', 'wonderland');

    await page.goto('/upload');
    await expect(page.getByRole('heading', { name: 'Upload a book' })).toBeVisible();

    await page.getByLabel('Title').fill('Playwright Sample');
    await page.getByLabel('Author').fill('Test Author');
    await page.getByLabel('Year').fill('2024');
    await page.getByLabel('Subject').fill('Integration Testing');
    await page.setInputFiles('input[type="file"]', sampleDocument);
    await page.getByRole('button', { name: 'Upload' }).click();

    await expect(
      page
        .getByRole('status')
        .filter({ hasText: /Upload completed successfully|Upload progress|Upload finished/i }),
    ).toBeVisible({ timeout: 60_000 });

    const bookId = (await page.locator('[data-testid="upload-result-book-id"]').innerText()).trim();
    expect(bookId).not.toEqual('—');

    await page.getByRole('button', { name: /Index now/i }).click();
    await expect(
      page
        .getByRole('status')
        .filter({ hasText: /Indexing started|Indexing…/i }),
    ).toBeVisible();

    await page.goto('/chat');
    await page.waitForSelector('.chat-session-list');
    await page.getByRole('button', { name: 'New Session' }).click();
    await expect(page.locator('.chat-session-item--active')).toBeVisible();

    const promptField = page.locator('#chat-prompt');
    await expect(promptField).toBeEnabled();

    const question = 'این کتاب درباره چیست؟';
    await promptField.fill(question);
    await page.getByRole('button', { name: 'Send' }).click();

    const assistantMessage = page.locator('[data-testid="chat-message-assistant"]').last();
    await expect(assistantMessage.locator('.chat-message__content')).toContainText(
      'Mock response generated for testing.',
      { timeout: 120_000 },
    );

    const citationLink = assistantMessage.locator('.chat-message__citations .chat-citation').first();
    await expect(citationLink).toBeVisible();
    await expect(citationLink).toContainText(bookId);
    const citationText = (await citationLink.textContent()) ?? '';
    const pageMatch = citationText.match(/:(\d+)/);

    await Promise.all([
      page.waitForURL(/\/viewer\?book=/),
      citationLink.click(),
    ]);

    await expect(page.getByRole('heading', { name: 'Book Viewer' })).toBeVisible();
    if (pageMatch) {
      await expect(page).toHaveURL(new RegExp(`#page=${pageMatch[1]}`));
    }

    const viewerUrl = new URL(page.url(), baseURL);
    const viewerPath = `${viewerUrl.pathname}${viewerUrl.search}${viewerUrl.hash}`;

    const secondContext = await browser.newContext({ baseURL });
    const secondPage = await secondContext.newPage();
    await performLogin(secondPage, 'bob@example.com', 'builder');
    await secondPage.goto(viewerPath);

    await expect(
      secondPage
        .getByRole('alert')
        .filter({ hasText: /(do not have access|Failed to load viewer)/i }),
    ).toBeVisible();

    await secondContext.close();
  });
});
