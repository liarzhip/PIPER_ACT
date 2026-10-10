import re
import tempfile
import unittest
from pathlib import Path
from web_app import create_app


class FrontendBuildTest(unittest.TestCase):
    def test_built_pages_assets_and_api(self):
        self.assertTrue((Path(__file__).resolve().parents[1] / "frontend/dist/index.html").is_file(), "Run npm run build in frontend first")
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(directory)
            client = app.test_client()
            for page in ("/", "/review.html"):
                response = client.get(page)
                self.assertEqual(response.status_code, 200)
                assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', response.text)
                self.assertGreaterEqual(len(assets), 2)
                response.close()
                for asset in assets:
                    resource = client.get(asset)
                    self.assertEqual(resource.status_code, 200, asset)
                    resource.close()
            self.assertEqual(client.get("/api/status").status_code, 200)
            app.extensions["recorder"].disconnect()