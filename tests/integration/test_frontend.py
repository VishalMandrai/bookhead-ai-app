# ─────────────────────────────────────────────────────────────────────────────
# tests/integration/test_frontend.py
#
# Integration tests for the frontend routes.
# Verifies that index.html is served, static assets are reachable,
# and the health check endpoint responds correctly.
# ─────────────────────────────────────────────────────────────────────────────

import pytest
from fastapi.testclient import TestClient


class TestHealthCheck:

    def test_health_returns_200(self, client):
        """GET /health must return HTTP 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client):
        """Health payload must contain status: ok."""
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_returns_version(self, client):
        """Health payload must include a version field."""
        resp = client.get("/health")
        assert "version" in resp.json()


class TestRootRoute:

    def test_root_returns_200(self, client):
        """GET / must return HTTP 200 (serves index.html or fallback JSON)."""
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_returns_html_or_json(self, client):
        """Root must return either HTML or JSON content-type."""
        resp = client.get("/")
        ct = resp.headers.get("content-type", "")
        assert "html" in ct or "json" in ct


class TestStaticFiles:

    def test_css_file_served(self, client):
        """The main CSS file must be accessible at /static/css/main.css."""
        resp = client.get("/static/css/main.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers.get("content-type", "")

    def test_js_app_served(self, client):
        """app.js must be accessible at /static/js/app.js."""
        resp = client.get("/static/js/app.js")
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "javascript" in ct or "text" in ct

    def test_js_api_served(self, client):
        resp = client.get("/static/js/api.js")
        assert resp.status_code == 200

    def test_js_ui_served(self, client):
        resp = client.get("/static/js/ui.js")
        assert resp.status_code == 200

    def test_js_reader_served(self, client):
        resp = client.get("/static/js/reader.js")
        assert resp.status_code == 200

    def test_js_librarian_served(self, client):
        resp = client.get("/static/js/librarian.js")
        assert resp.status_code == 200

    def test_nonexistent_static_returns_404(self, client):
        """A missing static file must return 404."""
        resp = client.get("/static/does-not-exist.xyz")
        assert resp.status_code == 404


class TestOpenAPISchema:

    def test_openapi_json_served(self, client):
        """FastAPI must serve the OpenAPI schema at /openapi.json."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "openapi" in schema
        assert "paths" in schema

    def test_docs_ui_served(self, client):
        """Swagger UI must be accessible at /docs."""
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_reader_routes_in_schema(self, client):
        """Reader API routes must appear in the OpenAPI schema."""
        schema = client.get("/openapi.json").json()
        paths = schema.get("paths", {})
        reader_paths = [p for p in paths if p.startswith("/reader")]
        assert len(reader_paths) >= 2

    def test_librarian_routes_in_schema(self, client):
        """Librarian API routes must appear in the OpenAPI schema."""
        schema = client.get("/openapi.json").json()
        paths = schema.get("paths", {})
        lib_paths = [p for p in paths if p.startswith("/librarian")]
        assert len(lib_paths) >= 3
