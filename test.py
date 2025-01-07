from flask import Flask, render_template, request, jsonify
from serpapi import GoogleSearch

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    data = request.json
    image_url = data.get('url')

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": "ef1060959fb01ad0a7d0000ed737a785872acf6d6b17b12ee71ef7b575e88999"
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
                "price": str(match.get("price", {}).get("currency", "")) + " " + str(match.get("price", {}).get("extracted_value", "N/A")),
                "thumbnail": match.get("thumbnail")
            })

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Image Outfit Analyzer</title>
#     <script>
#         async function analyzeImage() {
#             const imageUrl = document.getElementById('imageUrl').value;
#             if (!imageUrl) {
#                 alert('Please enter an image URL');
#                 return;
#             }

#             const response = await fetch('/analyze', {
#                 method: 'POST',
#                 headers: {
#                     'Content-Type': 'application/json',
#                 },
#                 body: JSON.stringify({ url: imageUrl }),
#             });

#             const data = await response.json();
#             const resultDiv = document.getElementById('results');
#             resultDiv.innerHTML = '';

#             if (data.error) {
#                 resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
#             } else {
#                 data.forEach((item) => {
#                     const resultItem = document.createElement('div');
#                     resultItem.innerHTML = `
#                         <h3>${item.title}</h3>
#                         <p>Price: ${item.price}</p>
#                         <a href="${item.link}" target="_blank">Buy Here</a>
#                         <br>
#                         <img src="${item.thumbnail}" alt="${item.title}" style="width: 100px; height: 100px;">
#                     `;
#                     resultDiv.appendChild(resultItem);
#                 });
#             }
#         }
#     </script>
# </head>
# <body>

#     <h1>Outfit Analyzer</h1>
#     <label for="imageUrl">Enter Image URL:</label>
#     <input type="text" id="imageUrl" placeholder="Enter Image URL">
#     <button onclick="analyzeImage()">Analyze</button>

#     <div id="results"></div>

# </body>
# </html>
