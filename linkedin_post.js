// linkedin_post.js
// Reads today's group names, generates a Haiku caption, screenshots the atlas,
// and posts to LinkedIn via Puppeteer.

const puppeteer = require('puppeteer');
const Anthropic = require('@anthropic-ai/sdk');
const fs = require('fs');

const ATLAS_URL = 'https://airesearchatlas.com';
const SCREENSHOT_PATH = '/tmp/atlas_screenshot.png';
const GROUPS_PATH = 'docs/data/groups.json';

// ─── 1. GENERATE CAPTION ────────────────────────────────────────────────────

async function generateCaption(groups) {
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

  // Use up to 8 group names as context
  const topGroups = groups.slice(0, 8).join(', ');

  const response = await client.messages.create({
    model: 'claude-haiku-4-5-20251001',
    max_tokens: 200,
    messages: [{
      role: 'user',
      content:
        `Today's AI Research Atlas covers these research clusters: ${topGroups}.\n\n` +
        `Write a single punchy LinkedIn sentence (max 30 words) that highlights what's ` +
        `trending in AI research today. Sound like a curious practitioner sharing a ` +
        `genuine observation — not a press release. End with: ${ATLAS_URL}\n\n` +
        `Return ONLY the sentence, nothing else.`
    }]
  });

  return response.content[0].text.trim();
}

// ─── 2. SCREENSHOT ──────────────────────────────────────────────────────────

async function takeScreenshot() {
  console.log('Launching browser for screenshot...');
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--enable-webgl',
      '--use-gl=swiftshader',         // software WebGL — no GPU needed in CI
      '--ignore-gpu-blocklist',
      '--window-size=1400,900',
    ]
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  console.log(`Loading ${ATLAS_URL}...`);
  await page.goto(ATLAS_URL, { waitUntil: 'networkidle2', timeout: 60000 });

  // Give the WebGL canvas time to render dots and labels
  console.log('Waiting 10s for canvas to render...');
  await delay(10000);

  await page.screenshot({ path: SCREENSHOT_PATH, type: 'png' });
  console.log(`Screenshot saved to ${SCREENSHOT_PATH}`);

  await browser.close();
}

// ─── 3. POST TO LINKEDIN ────────────────────────────────────────────────────

async function postToLinkedIn(caption, screenshotPath) {
  console.log('Launching browser for LinkedIn...');
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
    ]
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  page.setDefaultTimeout(30000);

  // ── Login ──────────────────────────────────────────────────────────────────
  console.log('Navigating to LinkedIn login...');
  await page.goto('https://www.linkedin.com/login', { waitUntil: 'networkidle2' });

  await page.type('#username', process.env.LINKEDIN_EMAIL, { delay: 60 });
  await page.type('#password', process.env.LINKEDIN_PASSWORD, { delay: 60 });
  await page.click('button[type="submit"]');
  await page.waitForNavigation({ waitUntil: 'networkidle2' });
  console.log('Logged in.');

  // ── Feed ───────────────────────────────────────────────────────────────────
  await page.goto('https://www.linkedin.com/feed/', { waitUntil: 'networkidle2' });
  await delay(2000);

  // ── Open post composer ────────────────────────────────────────────────────
  console.log('Opening post composer...');
  const startPostBtn = await page.waitForSelector(
    'button.share-box-feed-entry__trigger, ' +
    'button[aria-label*="post"], ' +
    '.share-box-feed-entry__closed-share-box'
  );
  await startPostBtn.click();
  await delay(2000);

  // ── Type caption ──────────────────────────────────────────────────────────
  console.log('Typing caption...');
  const editor = await page.waitForSelector('.ql-editor[contenteditable="true"]');
  await editor.click();
  await page.keyboard.type(caption, { delay: 25 });
  await delay(1000);

  // ── Upload image ──────────────────────────────────────────────────────────
  console.log('Uploading screenshot...');

  // Click the image/photo button to trigger file input
  const imageBtn = await page.$(
    'button[aria-label*="photo"], button[aria-label*="image"], button[aria-label*="Photo"]'
  ).catch(() => null);

  if (imageBtn) {
    await imageBtn.click();
    await delay(1500);
  }

  // Set the file on the hidden input
  const fileInput = await page.waitForSelector('input[type="file"][accept*="image"]');
  await fileInput.uploadFile(screenshotPath);
  await delay(3000);

  // LinkedIn sometimes adds a "Next" step after image selection
  const nextBtn = await page.$('button[aria-label="Next"]').catch(() => null);
  if (nextBtn) {
    console.log('Clicking Next...');
    await nextBtn.click();
    await delay(2000);
  }

  // ── Submit ─────────────────────────────────────────────────────────────────
  console.log('Clicking Post...');
  const postBtn = await page.waitForSelector(
    'button.share-actions__primary-action, ' +
    'button[aria-label*="Post"], ' +
    'button[data-control-name="share.post"]'
  );
  await postBtn.click();
  await delay(5000);

  console.log('Posted successfully.');
  await browser.close();
}

// ─── UTILS ──────────────────────────────────────────────────────────────────

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ─── MAIN ───────────────────────────────────────────────────────────────────

async function main() {
  if (!fs.existsSync(GROUPS_PATH)) {
    throw new Error(`Groups file not found: ${GROUPS_PATH}`);
  }

  const groups = JSON.parse(fs.readFileSync(GROUPS_PATH, 'utf8'));
  console.log(`Found ${groups.length} groups: ${groups.join(', ')}`);

  console.log('\n── Generating caption ──');
  const caption = await generateCaption(groups);
  console.log('Caption:', caption);

  console.log('\n── Taking screenshot ──');
  await takeScreenshot();

  console.log('\n── Posting to LinkedIn ──');
  await postToLinkedIn(caption, SCREENSHOT_PATH);

  console.log('\n✓ Done.');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
