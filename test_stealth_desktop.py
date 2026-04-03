import asyncio
from playwright.async_api import async_playwright
import sys

async def test():
    async with async_playwright() as p:
        # Use standard Chromium instead of CDP for this test
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        url = 'https://www.facebook.com/ramy.jus/posts/pfbid0LrGiVSKCSKfSv2QDUWePv5D1qTKVuGZW4KUsjUHe85JSGpSexaS1tfRhBiVh4Nwyl'
        
        print(f"Navigating to {url}...")
        try:
            await page.goto(url, wait_until="networkidle", timeout=45000)
            await asyncio.sleep(5)
            
            title = await page.title()
            print(f"Page Title: {title}")
            
            # Try to close the login modal
            close_buttons = [
                'div[aria-label="Fermer"]', 'div[aria-label="Close"]', 
                'div[role="button"]:has-text("X")', 'i.x1lliihq'
            ]
            for selector in close_buttons:
                btn = page.locator(selector).first
                if await btn.count() > 0 and await btn.is_visible():
                    await btn.click()
                    print(f"Closed modal via {selector}")
                    await asyncio.sleep(2)
                    break
            
            # Take post-close screenshot
            await page.screenshot(path="/tmp/fb_after_close.png")
            print("Screenshot after close saved to /tmp/fb_after_close.png")
            
            # Check for "All Comments" or "Most Relevant"
            # In English/French/German/etc.
            filter_labels = ["Plus pertinents", "Most relevant", "Relevante Kommentare"]
            for label in filter_labels:
                btns = page.get_by_text(label)
                if await btns.count() > 0:
                    print(f"Found filter button with label: {label}")
                    await btns.first.click()
                    await asyncio.sleep(1)
                    # Try to select "All comments" (Toutes les réponses / All comments)
                    all_labels = ["Toutes les réponses", "All comments", "Alle Kommentare"]
                    for all_label in all_labels:
                        all_btn = page.get_by_text(all_label)
                        if await all_btn.count() > 0:
                            await all_btn.first.click()
                            print(f"Selected '{all_label}' filter.")
                            await asyncio.sleep(2)
                            break
                    break
            
            # Final count of visible comments
            comments = await page.locator('div[role="article"]').count()
            print(f"Visible comments (div article): {comments}")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test())
