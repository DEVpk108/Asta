from __future__ import annotations

import re
import time
import webbrowser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SpotifySetupStatus(str, Enum):
    COMPLETED = "completed"
    WAITING_FOR_USER = "waiting_for_user"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SpotifySetupResult:
    status: SpotifySetupStatus
    summary: str
    client_id: str | None = None
    redirect_uri: str | None = None
    requires_user: bool = False
    metadata: dict | None = None


class SpotifySetupOperator:
    """Repair the Spotify developer integration without handling credentials.

    Browser login, MFA and CAPTCHA remain user-only boundaries. Once the
    authenticated developer dashboard is available, the operator uses DOM
    automation to create/configure the app and capture its Client ID.
    """

    dashboard_url = "https://developer.spotify.com/dashboard"
    default_redirect_uri = "http://127.0.0.1:8765/callback"

    def __init__(
        self,
        *,
        config_path: str | Path = ".env",
        browser_factory=None,
        notify=None,
        timeout_seconds: float = 300.0,
    ):
        self.config_path = Path(config_path)
        self.browser_factory = browser_factory
        self.notify = notify
        self.timeout_seconds = max(30.0, float(timeout_seconds))

    def run(self) -> SpotifySetupResult:
        from core.config import update_local_environment

        redirect_uri = self._redirect_uri()
        self._announce(
            "Spotify needs initial developer setup. I’m opening the Spotify Developer Dashboard and will continue automatically after any required sign-in."
        )

        try:
            client_id = self._run_browser_setup(redirect_uri)
        except ImportError:
            webbrowser.open(self.dashboard_url, new=2, autoraise=True)
            return SpotifySetupResult(
                status=SpotifySetupStatus.WAITING_FOR_USER,
                summary=(
                    "The Spotify Developer Dashboard was opened, but browser automation "
                    "is not installed. Install the optional browser requirements and "
                    "restart A.S.T.A."
                ),
                redirect_uri=redirect_uri,
                requires_user=True,
                metadata={"browser_automation": "missing"},
            )
        except _UserActionRequired as exc:
            return SpotifySetupResult(
                status=SpotifySetupStatus.WAITING_FOR_USER,
                summary=str(exc),
                redirect_uri=redirect_uri,
                requires_user=True,
                metadata={"browser_automation": "waiting_for_user"},
            )
        except Exception as exc:
            return SpotifySetupResult(
                status=SpotifySetupStatus.FAILED,
                summary=f"Spotify setup failed: {type(exc).__name__}: {exc}",
                redirect_uri=redirect_uri,
                metadata={"browser_automation": "error"},
            )

        if not client_id:
            return SpotifySetupResult(
                status=SpotifySetupStatus.FAILED,
                summary="Spotify setup completed without a Client ID.",
                redirect_uri=redirect_uri,
            )

        update_local_environment(
            {
                "ASTA_SPOTIFY_CLIENT_ID": client_id,
                "ASTA_SPOTIFY_REDIRECT_URI": redirect_uri,
            },
            self.config_path,
        )

        self._announce(
            "Spotify developer setup is complete. I’m resuming the original task."
        )
        return SpotifySetupResult(
            status=SpotifySetupStatus.COMPLETED,
            summary="Spotify developer app configured successfully.",
            client_id=client_id,
            redirect_uri=redirect_uri,
        )

    def _run_browser_setup(self, redirect_uri: str) -> str:
        factory = self.browser_factory
        if factory is None:
            factory = _PlaywrightSpotifyBrowser

        browser = factory(
            dashboard_url=self.dashboard_url,
            redirect_uri=redirect_uri,
            notify=self._announce,
            timeout_seconds=self.timeout_seconds,
        )
        return browser.run()

    def _redirect_uri(self) -> str:
        import os

        return (
            os.getenv("ASTA_SPOTIFY_REDIRECT_URI", self.default_redirect_uri).strip()
            or self.default_redirect_uri
        )

    def _announce(self, message: str) -> None:
        if callable(self.notify):
            self.notify(message)


class _UserActionRequired(RuntimeError):
    pass


class _PlaywrightSpotifyBrowser:
    """Playwright adapter kept isolated so unit tests do not require a browser."""

    def __init__(
        self,
        *,
        dashboard_url: str,
        redirect_uri: str,
        notify=None,
        timeout_seconds: float = 300.0,
    ):
        self.dashboard_url = dashboard_url
        self.redirect_uri = redirect_uri
        self.notify = notify
        self.timeout_seconds = timeout_seconds

    def _announce(self, message: str) -> None:
        if callable(self.notify):
            self.notify(message)

    def run(self) -> str:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright

        from os import getenv

        profile = Path(
            getenv(
                "ASTA_SPOTIFY_BROWSER_PROFILE",
                "data/browser/spotify",
            )
        ).resolve()
        profile.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as playwright:
            channel = getenv("ASTA_BROWSER_CHANNEL", "chrome").strip()
            try:
                context = playwright.chromium.launch_persistent_context(
                    str(profile),
                    channel=channel or None,
                    headless=False,
                )
            except Exception as first_exc:
                print(
                    f"[Setup/Spotify] Browser launch via channel "
                    f"{channel or '<default>'} failed: "
                    f"{type(first_exc).__name__}: {first_exc}",
                    flush=True,
                )
                executable = self._find_windows_chrome()
                if executable is None:
                    raise

                print(
                    f"[Setup/Spotify] Retrying browser launch with executable: "
                    f"{executable}",
                    flush=True,
                )
                context = playwright.chromium.launch_persistent_context(
                    str(profile),
                    executable_path=executable,
                    headless=False,
                )
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(8_000)
            page.goto(self.dashboard_url, wait_until="domcontentloaded")
            page.bring_to_front()

            # Spotify renders the dashboard client-side; allow a bounded
            # hydration window before deciding a control is unavailable.
            self._wait_for_dashboard_render(page)

            if self._login_visible(page):
                self._announce(
                    "Please sign in to your Spotify account in the A.S.T.A. browser window. I’ll continue after the Developer Dashboard becomes available."
                )
                if not self._wait_for_dashboard(page, PlaywrightTimeoutError):
                    context.close()
                    raise _UserActionRequired(
                        "Spotify sign-in is required. Finish sign-in in the A.S.T.A. browser window; the task remains paused until setup can continue."
                    )

            if self._terms_gate_visible(page):
                self._announce(
                    "Spotify is asking for its latest Developer Terms before I can create the app. Please complete that account-level step in the browser."
                )
                if not self._wait_for_dashboard(page, PlaywrightTimeoutError):
                    context.close()
                    raise _UserActionRequired(
                        "Spotify Developer Terms setup is still required in the browser."
                    )

            self._open_or_create_app(page)

            if not self._redirect_configured(page):
                self._configure_redirect(page)

            client_id = self._extract_client_id(page)
            context.close()
            if not client_id:
                raise RuntimeError(
                    "Could not locate the Spotify Client ID after app setup."
                )
            return client_id

    @staticmethod
    def _find_windows_chrome() -> str | None:
        import os

        candidates = [
            os.getenv("PROGRAMFILES", ""),
            os.getenv("PROGRAMFILES(X86)", ""),
            os.getenv("LOCALAPPDATA", ""),
        ]
        suffix = Path("Google") / "Chrome" / "Application" / "chrome.exe"
        for root in candidates:
            if not root:
                continue
            candidate = Path(root) / suffix
            if candidate.exists():
                return str(candidate)
        return None

    def _open_or_create_app(self, page) -> None:
        existing = page.get_by_text(
            re.compile(r"^A\.S\.T\.A\.?$", re.IGNORECASE)
        )
        if existing.count():
            existing.first.click()
            page.wait_for_load_state("domcontentloaded")
            return

        button = self._find_create_app_control(page)
        if button is None:
            url = str(getattr(page, "url", "") or "")
            body = self._safe_body_text(page)
            hold_markers = (
                "new integrations are currently on hold",
                "temporarily unavailable",
                "unable to create",
                "app creation is currently unavailable",
            )
            if any(marker in body.lower() for marker in hold_markers):
                raise _UserActionRequired(
                    "Spotify is currently not allowing app creation on this developer account. "
                    "Please resolve the account/dashboard restriction in the browser; A.S.T.A. will resume afterward."
                )
            print(
                "[Setup/Spotify] Create App control not found after render wait. "
                f"url={url or '<unknown>'} body={body[:1200]!r}",
                flush=True,
            )
            raise RuntimeError(
                "Spotify Developer Dashboard did not expose a Create App control after the page rendered."
            )

        button.click()

        self._fill(page, re.compile(r"app\s*name|name", re.IGNORECASE), "A.S.T.A.")
        self._fill(
            page,
            re.compile(r"app\s*description|description", re.IGNORECASE),
            "A.S.T.A. local-first AI engineering assistant.",
        )

        self._click_text_if_visible(page, re.compile(r"Web API", re.IGNORECASE))

        if self._terms_checkbox_visible(page):
            self._announce(
                "Spotify requires Developer Terms acceptance before the app can be created. Please accept the terms in the A.S.T.A. browser window; I’ll continue automatically afterward."
            )
            if not self._wait_for_terms_acceptance(page):
                raise _UserActionRequired(
                    "Spotify Developer Terms still need to be accepted in the browser."
                )

        create = self._first_visible(
            page,
            (
                re.compile(r"^create$", re.IGNORECASE),
                re.compile(r"create\s+app", re.IGNORECASE),
            ),
            role="button",
        )
        if create is None:
            raise RuntimeError("Spotify app creation dialog has no Create control.")
        create.click()
        page.wait_for_load_state("domcontentloaded")
        time.sleep(0.5)

    def _wait_for_dashboard_render(self, page) -> None:
        deadline = time.monotonic() + min(15.0, self.timeout_seconds)
        while time.monotonic() < deadline:
            if self._login_visible(page):
                return
            if self._find_create_app_control(page) is not None:
                return
            try:
                if "dashboard" in str(page.url).lower():
                    body = self._safe_body_text(page).lower()
                    if "developer terms" in body:
                        return
            except Exception:
                pass
            try:
                page.wait_for_timeout(500)
            except Exception:
                time.sleep(0.5)

    def _find_create_app_control(self, page):
        patterns = (
            re.compile(r"^create\s+app$", re.IGNORECASE),
            re.compile(r"^create\s+an?\s+app$", re.IGNORECASE),
            re.compile(r"create\s+app", re.IGNORECASE),
            re.compile(r"create\s+an?\s+app", re.IGNORECASE),
        )

        # Depending on the current dashboard layout, the CTA may expose a
        # button or link accessibility role.
        for role in ("button", "link"):
            for pattern in patterns:
                try:
                    locator = page.get_by_role(role, name=pattern)
                    if locator.count() and locator.first.is_visible():
                        return locator.first
                except Exception:
                    pass

        try:
            locator = page.get_by_text(
                re.compile(r"^create\s+app$|^create\s+an?\s+app$", re.IGNORECASE)
            )
            if locator.count() and locator.first.is_visible():
                return locator.first
        except Exception:
            pass

        # Last-resort scan of visible buttons/links and their accessible text.
        try:
            candidates = page.locator("button, a, [role='button'], [role='link']")
            for index in range(min(candidates.count(), 64)):
                candidate = candidates.nth(index)
                if not candidate.is_visible():
                    continue
                try:
                    label = " ".join((
                        candidate.inner_text(),
                        candidate.get_attribute("aria-label") or "",
                        candidate.get_attribute("title") or "",
                    )).strip()
                except Exception:
                    continue
                if re.search(r"\bcreate\s+(?:an?\s+)?app\b", label, re.IGNORECASE):
                    return candidate
        except Exception:
            pass

        return None

    @staticmethod
    def _safe_body_text(page) -> str:
        try:
            return str(page.locator("body").inner_text() or "").strip()
        except Exception:
            return ""

    def _configure_redirect(self, page) -> None:
        settings = self._first_visible(
            page,
            (
                re.compile(r"edit\s+settings", re.IGNORECASE),
                re.compile(r"settings", re.IGNORECASE),
            ),
            role="button",
        )
        if settings is not None:
            settings.click()
            time.sleep(0.25)

        self._click_text_if_visible(page, re.compile(r"edit\s+settings", re.IGNORECASE))

        field = self._first_visible(
            page,
            (re.compile(r"redirect\s*uri", re.IGNORECASE),),
            role=None,
        )
        if field is None:
            inputs = page.locator("input")
            for index in range(min(inputs.count(), 24)):
                candidate = inputs.nth(index)
                try:
                    if not candidate.is_visible():
                        continue
                    value = candidate.input_value()
                    if self.redirect_uri == value:
                        return
                    if not value:
                        field = candidate
                        break
                except Exception:
                    continue

        if field is not None:
            field.fill(self.redirect_uri)
        else:
            add = self._first_visible(
                page,
                (
                    re.compile(r"add\s+(redirect\s+uri|uri)", re.IGNORECASE),
                ),
                role=("button",),
            )
            if add is not None:
                add.click()
                field = self._first_visible(
                    page,
                    (re.compile(r"redirect\s*uri", re.IGNORECASE),),
                    role=None,
                )
                if field is not None:
                    field.fill(self.redirect_uri)

        save = self._first_visible(
            page,
            (re.compile(r"^save$", re.IGNORECASE),),
            role="button",
        )
        if save is not None:
            save.click()
            time.sleep(0.5)

    def _redirect_configured(self, page) -> bool:
        body = page.locator("body").inner_text()
        return self.redirect_uri in body

    def _extract_client_id(self, page) -> str | None:
        body = page.locator("body").inner_text()
        match = re.search(
            r"client\s*id\s*[:\n\s]+([A-Za-z0-9]{20,64})",
            body,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)

        inputs = page.locator("input")
        for index in range(min(inputs.count(), 64)):
            field = inputs.nth(index)
            try:
                value = field.input_value().strip()
            except Exception:
                continue
            if re.fullmatch(r"[A-Za-z0-9]{20,64}", value):
                return value

        return None

    def _terms_checkbox_visible(self, page) -> bool:
        body = page.locator("body").inner_text().lower()
        if "developer terms" not in body:
            return False
        try:
            return page.locator("input[type='checkbox']").count() > 0
        except Exception:
            return False

    def _wait_for_terms_acceptance(self, page) -> bool:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            try:
                checkboxes = page.locator("input[type='checkbox']")
                if checkboxes.count() == 0:
                    return True

                if any(
                    checkboxes.nth(index).is_checked()
                    for index in range(min(checkboxes.count(), 4))
                ):
                    return True

                page.wait_for_timeout(750)
            except Exception:
                pass

        return False

    def _login_visible(self, page) -> bool:
        url = str(page.url).lower()
        body = page.locator("body").inner_text().lower()
        try:
            credential_fields = page.locator(
                "input[type='email'], input[type='password']"
            ).count()
        except Exception:
            credential_fields = 0

        return (
            "accounts.spotify.com/login" in url
            or "log in to spotify" in body
            or credential_fields > 0
        )

    def _terms_gate_visible(self, page) -> bool:
        body = page.locator("body").inner_text().lower()
        try:
            checkbox_count = page.locator("input[type='checkbox']").count()
        except Exception:
            checkbox_count = 0
        return (
            "developer terms" in body
            and ("accept" in body or "agree" in body)
            and checkbox_count > 0
            and "dashboard" not in str(page.url).lower()
        )

    def _wait_for_dashboard(self, page, timeout_error_type) -> bool:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            try:
                if not self._login_visible(page) and not self._terms_gate_visible(page):
                    return True
                page.wait_for_timeout(1000)
            except timeout_error_type:
                pass
        return False

    @staticmethod
    def _first_visible(page, patterns, *, role=None):
        for pattern in patterns:
            locator = (
                page.get_by_role(role, name=pattern)
                if role
                else page.get_by_label(pattern)
            )
            try:
                if locator.count() and locator.first.is_visible():
                    return locator.first
            except Exception:
                pass

            if role:
                locator = page.get_by_text(pattern)
                try:
                    if locator.count() and locator.first.is_visible():
                        return locator.first
                except Exception:
                    pass

        return None

    @staticmethod
    def _fill(page, pattern, value):
        field = page.get_by_label(pattern)
        if field.count() and field.first.is_visible():
            field.first.fill(value)
            return
        inputs = page.locator("input")
        for index in range(min(inputs.count(), 24)):
            candidate = inputs.nth(index)
            try:
                if candidate.is_visible() and not candidate.input_value():
                    candidate.fill(value)
                    return
            except Exception:
                continue
        raise RuntimeError(f"Spotify setup form field not found: {pattern.pattern}")

    @staticmethod
    def _click_text_if_visible(page, pattern) -> bool:
        locator = page.get_by_text(pattern)
        try:
            if locator.count() and locator.first.is_visible():
                locator.first.click()
                return True
        except Exception:
            return False
        return False


def _first_visible(page, patterns, *, role=None):
    return _PlaywrightSpotifyBrowser._first_visible(page, patterns, role=role)
