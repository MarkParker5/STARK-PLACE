import { expect, test } from "@playwright/test";

// Catches build/runtime breakage from front-end dep bumps: the bundle must mount,
// render its shell, load the bundled sample trace, and raise no uncaught exceptions.
// (The /api/* fetches fail without a backend and are caught — that's expected, so we
// assert on uncaught page errors, not console noise.)
test("visualizer mounts and renders the sample trace", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (e) => pageErrors.push(String(e)));

  await page.goto("/", { waitUntil: "networkidle" });

  await expect(page).toHaveTitle(/Visualizer/i);
  await expect(page.getByText("VISUALIZER")).toBeVisible();
  await expect(page.getByText("Dashboard").first()).toBeVisible();
  // data-derived content: only rendered once the sample trace is fetched + parsed
  await expect(page.getByText("CommandsContext").first()).toBeVisible({ timeout: 15_000 });

  expect(pageErrors, `uncaught page errors:\n${pageErrors.join("\n")}`).toEqual([]);
});
