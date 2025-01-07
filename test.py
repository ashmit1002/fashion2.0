from flask import Flask, request, jsonify
from serpapi import GoogleSearch

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    data = request.json
    image_url = data.get('url')

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": "YOUR_SERPAPI_KEY"
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        visual_matches = results.get("visual_matches", [])
        top_matches = visual_matches[:3]

        response = []
        for match in top_matches:
            response.append({
                "title": match.get("title"),
                "link": match.get("link"),
                "price": match.get("price", {}).get("value", "N/A"),
                "thumbnail": match.get("thumbnail")
            })

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
