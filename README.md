# Feedback Fusion - A Multiscope Sentiment Analysis Tool 🌟

## Overview 📜
Feedback Fusion is a comprehensive sentiment analysis tool designed to analyze customer feedback across multiple platforms, including Amazon, Flipkart, Meesho, Myntra, and YouTube. The tool scrapes reviews and comments from these platforms, performs sentiment analysis using NLTK's VADER and Logistic Regression, and provides interactive visualizations to help users understand customer sentiments.

The project is built using Streamlit, making it easy to deploy and use as a web application. It integrates web scraping, natural language processing (NLP), and data visualization to deliver actionable insights from customer feedback.

## Features 🚀
- **Cross-Platform Review Scraping**: Fetch reviews from Amazon, Flipkart, Meesho, Myntra, and YouTube using product URLs or video IDs. 🌐
- **Sentiment Analysis**: Classify reviews into Positive, Negative, and Neutral categories using NLTK's VADER and Logistic Regression. 📊
- **Interactive Visualizations**:
  - **Bar Charts**: Display sentiment distribution. 📈
  - **Pie Charts**: Show sentiment percentages. 🥧
  - **Scatter Plots**: Visualize sentiment scores. ✨
  - **Histograms**: Analyze sentiment score distribution with density curves. 📉
- **User-Friendly Interface**: Built with Streamlit, the app provides an intuitive and interactive experience. 🖥️
- **Real-Time Analytics**: Get instant insights into customer feedback. ⏱️

## Installation 🛠️
To run Feedback Fusion locally, follow these steps:

### Clone the Repository:
```bash
git clone https://github.com/your-username/feedback-fusion.git
cd feedback-fusion
```

### Set Up a Virtual Environment (Optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Download NLTK Data:
```bash
python -m nltk.downloader vader_lexicon
```

### Run the Application:
```bash
streamlit run app.py
```

### Access the App:
Open your browser and navigate to [http://localhost:8501](http://localhost:8501).

## Usage 🎮
### Platform Selection:
- Choose a platform (YouTube, Amazon, Flipkart, Meesho, or Myntra) from the sidebar. 🎯
- Enter the product URL or YouTube video ID. 🔗

### Analyze Reviews:
- Click the **"Analyze Reviews"** or **"Analyze Comments"** button to fetch and analyze reviews/comments. 🔍

### View Results:
- The app will display the reviews/comments along with their sentiment classification. 📝
- Interactive visualizations (bar charts, pie charts, scatter plots, and histograms) will provide insights into sentiment distribution. 📊

### Explore Technical Details:
- Navigate to the **"Technical Details"** section in the sidebar to learn about the tools and libraries used in the project. 🛠️

## Technical Stack 💻
- **Web Framework**: Streamlit 🚀
- **Data Handling**: pandas, numpy 🐼
- **Natural Language Processing**: NLTK, VADER Sentiment Analysis 📚
- **Machine Learning**: scikit-learn (Logistic Regression, TfidfVectorizer) 🤖
- **Web Scraping**: BeautifulSoup, Selenium 🕷️
- **Visualization**: matplotlib, seaborn, plotly 📊
- **APIs**: YouTube Data API v3 📹

## Project Structure 🗂️
```bash
feedback-fusion/
├── app.py                  # Main Streamlit application
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
├── sentiment_data.csv      # Sample dataset for sentiment analysis
├── images/                 # Folder for platform logos (Amazon, Flipkart, etc.)
│   ├── amazon.jpg
│   ├── flipkart.jpg
│   ├── meesho.jpg
│   ├── myntra.jpg
│   └── youtube.jpg
```

## Contributing 🤝
Contributions are welcome! If you'd like to contribute to Feedback Fusion, please follow these steps:

1. **Fork** the repository. 🍴
2. **Create a new branch** for your feature or bugfix. 🌿
3. **Commit** your changes. 💾
4. **Submit a pull request**. 🔄

## License 📜
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏
- **NLTK** for providing the VADER sentiment analysis tool. 📚
- **Streamlit** for making it easy to build and deploy interactive web apps. 🚀
- **Plotly and Matplotlib** for enabling rich and interactive visualizations. 📊

## Contact 📧
For questions or feedback, feel free to reach out:

- **Gmail**: prathyushavanama215@gmail.com 📩
- **GitHub**: https://github.com/Prathyusha-215 🐙

Feedback Fusion is your one-stop solution for understanding customer sentiments across multiple platforms. Give it a try and unlock the power of feedback! 🚀✨
