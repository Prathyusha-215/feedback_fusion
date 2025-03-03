import streamlit as st
from streamlit_extras.colored_header import colored_header
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager






# Download NLTK VADER lexicon if not already downloaded
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Set the title and configure the page
st.set_page_config(page_title="Feedback Fusion", page_icon="🌀", layout="wide")

# Background style customization and font color adjustment
page_bg = """
<style>
.stApp {
    background: linear-gradient(135deg, #d4e4f7, #f0f4fa);
    padding: 10px;
    transition: all 0.3s ease-in-out;
}
h1 {
    color: #00aaff;
    font-size: 3.5em;
    text-align: center;
    font-weight: bold;
    margin-bottom: 20px;
    font-family: 'Arial', sans-serif;
    animation: fadeIn 2s ease-in-out;
}
h2, h3, p {
    color: #333;
    text-align: center;
    font-family: 'Verdana', sans-serif;
    animation: slideIn 1.5s ease-in-out;
}

/* Center the button and add hover effect */
div.stButton > button {
    background-color: #00aaff;
    color: white;
    font-size: 1.3em;
    font-weight: bold;
    padding: 12px 25px;
    border-radius: 12px;
    border: 2px solid #0077cc;
    margin: 0 auto;
    display: block;
    box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
    transition: background-color 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
}

/* Button hover effect */
div.stButton > button:hover {
    background-color: #0077cc;
    color: #fff;
    box-shadow: 5px 5px 12px rgba(0, 0, 0, 0.3);
    cursor: pointer;
}

/* Hover effects for logos */
img:hover {
    transform: scale(1.1);
    transition: transform 0.3s ease-in-out;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideIn {
    from { transform: translateY(30px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Initialize session state variable if it doesn't exist
if 'started' not in st.session_state:
    st.session_state.started = False

# Main content display logic for the initial page
if not st.session_state.started:
    # Main title with emoji
    st.markdown("<h1>🌀 Feedback Fusion 🌀</h1>", unsafe_allow_html=True)

    # Subheader with an engaging intro
    st.subheader("Your one-stop solution for cross-platform review analysis")

    # Create colored sections using the streamlit_extras library for added emphasis
    colored_header("Welcome!", color_name="light-blue-70")

    st.write("""
    At **Feedback Fusion**, we bring together reviews from your favorite shopping platforms like **Amazon**, **Meesho**, **Myntra**, **Flipkart**, and **YouTube**. 
    We analyze customer sentiments to give you actionable insights into what people are really saying about products.

    ### Here's what you can explore:
    -**Seamless Scraping**: Fetch reviews from product pages with just a URL.\n
    -**Sentiment Insights**: Break down customer feedback into **Positive**, **Negative**, and **Neutral** categories.\n
    -**Real-time Analytics**: Get a quick snapshot of the sentiment distribution.
    """)

    # Add columns for a more engaging layout, ensuring logos are in a single row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    logo_width = 100  # Set the desired width for the logos

    with col1:
        st.image('amazon.jpg', caption='Amazon', width=logo_width)
    with col2:
        st.image('myntra.jpg', caption='Myntra', width=logo_width)
    with col3:
        st.image('flipkart.jpg', caption='Flipkart', width=logo_width)
    with col4:
        st.image('meesho.jpg', caption='Meesho', width=logo_width)
    with col5:
        st.image('youtube.jpg', caption='YouTube', width=logo_width)

    # Call to action
    st.markdown("<h2>Ready to unlock the power of feedback? 🚀</h2>", unsafe_allow_html=True)

    # Add a centered and highlighted button
    if st.button('Get Started'):
        st.session_state.started = True  # Set state to indicate that user has clicked "Get Started"
        #st.experimental_rerun()  # This line refreshes the app to display the new content

# If the user has clicked the button, display the new page content
if st.session_state.started:
    st.title("FeedbackFusion: Multi-Platform Sentiment Analysis")

    # Function to load the sentiment model
    def load_model():
        try:
            data = pd.read_csv('sentiment_data.csv')  # Your dataset
        except FileNotFoundError:
            data = pd.DataFrame({
                'text': [
                    "I love this!", "I hate this!", "Not bad", "Terrible service", "Great!", "Awful!",
                    "The product exceeded my expectations", "This is the worst I have seen",
                    "Amazing quality, highly recommend!", "Horrible experience, would never buy again",
                    "Fantastic value for money", "Very disappointed with the purchase",
                    "Excellent performance", "The customer service was terrible",
                    "Highly efficient and fast delivery", "Not worth the price at all",
                    "Perfect for daily use", "Completely unreliable product",
                    "Great product, highly recommend!", 
                    "Very disappointed with the quality.", 
                    "It works as expected, but the service was lacking.", 
                    "Terrible experience, I would not buy again.",
                    "Could be better, but it's okay for the price."
                ],
                'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,1, 0, 0, 0, 1]
            })

        tfidf = TfidfVectorizer(max_features=1000)
        X = tfidf.fit_transform(data['text'])
        model = LogisticRegression()
        model.fit(X, data['label'])
        return model, tfidf

    model, tfidf = load_model()

    def classify_sentiment(review_text):
        sentiment_scores = sid.polarity_scores(review_text)
        # Extract the compound score
        compound_score = sentiment_scores['compound']
        
        # Define thresholds for positive, neutral, and negative sentiment
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def plot_bar_chart(df):
        # Ensure 'sentiment' column exists for generating sentiment counts
        if 'sentiment' not in df.columns:
            st.error("DataFrame must contain a 'sentiment' column.")
            return

        # Count sentiment occurrences
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Define colors for each sentiment
        sentiment_color_map = {
            "Positive": "green",
            "Neutral": "orange",
            "Negative": "red"
        }
        # Map colors to the sentiment column
        sentiment_counts['Color'] = sentiment_counts['Sentiment'].map(sentiment_color_map)

        # Create interactive bar chart with Plotly
        fig = go.Figure()

        # Add bars
        fig.add_trace(go.Bar(
            x=sentiment_counts['Sentiment'],
            y=sentiment_counts['Count'],
            marker_color=sentiment_counts['Color'],  # Use the mapped colors for each sentiment
            text=sentiment_counts.apply(lambda x: f"Count: {x['Count']}<br>Percentage: {x['Count'] / len(df) * 100:.2f}%", axis=1),
            textposition='outside',  # Display the text above the bars
            name='Sentiment Count'
        ))

        # Add layout enhancements
        fig.update_layout(
            title='Interactive Bar Chart of Sentiment Counts',
            xaxis_title='Sentiment',
            yaxis_title='Count',
            template='plotly_dark',
            height=500,
            width=700,
            xaxis=dict(showgrid=False, tickmode='linear'),
            yaxis=dict(showgrid=True),
            hovermode='x',
            margin=dict(l=50, r=50, t=80, b=50),  # Adjust margins for better spacing
            bargap=0.2,  # Adjust gap between bars for better clarity
            transition=dict(duration=500)  # Smooth transitions for updates
        )

        # Add a dropdown for filtering specific sentiments dynamically
        fig.update_layout(
            updatemenus=[
                {
                    'buttons': [
                        {'method': 'restyle', 'label': 'All', 'args': [{'visible': [True, True, True]}]},
                        {'method': 'restyle', 'label': 'Positive', 'args': [{'visible': [True, False, False]}]},
                        {'method': 'restyle', 'label': 'Neutral', 'args': [{'visible': [False, True, False]}]},
                        {'method': 'restyle', 'label': 'Negative', 'args': [{'visible': [False, False, True]}]}
                    ],
                    'direction': 'down',
                    'pad': {"r": 10, "t": 10},
                    'showactive': True,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 1.2,
                    'yanchor': 'top'
                }
            ]
        )

        # Display the bar chart in Streamlit
        st.plotly_chart(fig)

    def plot_pie_chart(df):
        # Ensure 'sentiment' column exists for generating sentiment counts
        if 'sentiment' not in df.columns:
            st.error("DataFrame must contain a 'sentiment' column.")
            return

        # Count sentiment occurrences
        sentiment_counts = df['sentiment'].value_counts()

        # Define colors for each sentiment
        sentiment_color_map = {
            "Positive": "green",
            "Neutral": "orange",
            "Negative": "red"
        }

        # Create interactive pie chart with Plotly
        fig = px.pie(
            values=sentiment_counts,
            names=sentiment_counts.index,
            title='Interactive Pie Chart of Sentiment Distribution',
            color_discrete_map=sentiment_color_map,
            hole=0.4  # Creates a donut-style pie chart
        )

        # Add hover template for more information on hover
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>" +
                        "Count: %{value}<br>" +
                        "Percentage: %{percent}",
            textinfo='percent+label',  # Show both percentage and label in the chart
            pull=[0.05 if sentiment_counts.index[i] == "Positive" else 0.1 if sentiment_counts.index[i] == "Neutral" else 0.15
                for i in range(len(sentiment_counts))],  # Pull out slices for better visualization
            marker=dict(
                line=dict(color='#000000', width=2)  # Adds a black border for better contrast
            )
        )

        # Update layout with additional features
        fig.update_layout(
            template='plotly_dark',  # Use a dark theme for a modern look
            height=500,
            width=700,
            legend=dict(
                title='Sentiment Types',
                orientation="h",  # Horizontal legend at the bottom
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=50, r=50, t=80, b=50),  # Adjust margins for better spacing
            transition=dict(duration=500)  # Smooth transition for better interactivity
        )

        # Allow toggling visibility of slices from the legend
        fig.update_traces(
            selector=dict(type='pie'),
            pull=[0.05 if sentiment_counts.index[i] == "Positive" else 0.1 if sentiment_counts.index[i] == "Neutral" else 0.15
                for i in range(len(sentiment_counts))]
        )

        # Display the pie chart in Streamlit
        st.plotly_chart(fig)

    def plot_scatter_plot(df):
        # Ensure 'sentiment' column exists
        if 'sentiment' not in df.columns:
            st.error("DataFrame must contain a 'sentiment' column.")
            return

        # Map sentiment values to colors, adding a default for unknown values
        sentiment_color_map = {
            "Positive": "green",
            "Negative": "red",
            "Neutral": "orange"
        }
        
        # Map sentiment to colors and use 'gray' for any unknown types
        df['color'] = df['sentiment'].map(sentiment_color_map).fillna('gray')

        # Calculate sentiment scores if not already present
        df['Sentiment Score'] = df.get('Sentiment Score') or df['sentiment'].apply(lambda x: sid.polarity_scores(x)['compound'])
        
        # Create scatter plot
        fig = go.Figure()

        # Add scatter points
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Sentiment Score'],
            mode='markers',
            marker=dict(
                size=10,
                color=df['color'],  # Use the mapped colors
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            # Use placeholder text for hover if 'review' is not available
            text=df.get('review', "No review available"),
            hovertemplate="<b>Sentiment Score:</b> %{y}<br><b>Index:</b> %{x}<br><b>Review:</b> %{text}",
            name='Sentiment Score'
        ))

        # Add layout enhancements
        fig.update_layout(
            title='Interactive Scatter Plot of Sentiment Scores',
            xaxis_title='Review Index',
            yaxis_title='Sentiment Score',
            template='plotly_dark',
            height=500,
            width=700,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True),
            hovermode='closest'  # Hover displays the nearest point information
        )

        # Display the scatter plot in Streamlit
        st.plotly_chart(fig)

    def plot_histogram(df):
        # Ensure 'sentiment' column exists for generating sentiment scores
        if 'sentiment' not in df.columns:
            st.error("DataFrame must contain a 'sentiment' column.")
            return

        # Calculate sentiment scores if 'Sentiment Score' column is not already present
        if 'Sentiment Score' not in df.columns:
            df['Sentiment Score'] = df['sentiment'].apply(lambda x: sid.polarity_scores(x)['compound'])
        
        # Add small jitter to 'Sentiment Score' to avoid singular matrix issues
        df['Sentiment Score'] += np.random.normal(0, 0.01, size=len(df))

        # Check the variance of the 'Sentiment Score'
        sentiment_variance = np.var(df['Sentiment Score'])
        if sentiment_variance < 1e-6:
            st.warning("Variance in 'Sentiment Score' is too low for reliable KDE. Displaying histogram only.")
            kde_available = False
        else:
            kde_available = True

        # Create an interactive histogram with Plotly
        fig = go.Figure()

        # Add histogram
        fig.add_trace(go.Histogram(
            x=df['Sentiment Score'],
            nbinsx=30,  # More bins for finer granularity
            marker_color='#00CC96',  # Green color for bars
            opacity=0.7,  # Slight transparency for visual appeal
            name='Histogram'
        ))

        # Add density curve only if variance is sufficient
        if kde_available:
            sentiment_scores = df['Sentiment Score'].values
            kde = gaussian_kde(sentiment_scores, bw_method=0.3)  # Adjust bw_method for smoother curves
            x_vals = np.linspace(min(sentiment_scores), max(sentiment_scores), 100)
            density_vals = kde(x_vals)

            # Add density curve
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=density_vals * len(df) * (max(sentiment_scores) - min(sentiment_scores)) / 30,  # Scale density to match histogram
                mode='lines',
                line=dict(color='deepskyblue', width=2),
                fill='tozeroy',
                name='Density Curve'
            ))

        # Add layout enhancements
        fig.update_layout(
            title='Interactive Histogram of Sentiment Scores with Density Curve',
            xaxis_title='Sentiment Score',
            yaxis_title='Frequency',
            template='plotly_dark',  # Dark theme for modern look
            height=500,
            width=700,
            xaxis=dict(
                showgrid=False,  # Removes grid lines for a cleaner look
                tickmode='linear',  # Ensures consistent tick spacing
                tick0=-1,
                dtick=0.1
            ),
            yaxis=dict(showgrid=True),
            bargap=0.05,  # Minimizes gap between histogram bars for a smoother visual
            hovermode='x'  # Displays tooltips on hover
        )

        # Add sliders for dynamic bin adjustment
        fig.update_layout(
            sliders=[{
                'pad': {"t": 50},
                'x': 0.1,
                'len': 0.9,
                'active': 5,
                'steps': [
                    {'label': str(i), 'method': 'restyle', 'args': ['nbinsx', i]} 
                    for i in range(10, 60, 5)  # Adjust bins dynamically between 10 and 60
                ]
            }]
        )

        # Display the histogram in Streamlit
        st.plotly_chart(fig)
        


    def scrape_amazon_reviews(url):
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in headless mode
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        # Create the driver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        try:
            driver.get(url)
            time.sleep(2)  # Allow time for the page to load

            # Scroll to load more reviews
            for _ in range(3):  # Scroll multiple times to load more content
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(2, 4))  # Wait for new reviews to load

            # Get page source after scrolling
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            # List of possible selectors for Amazon reviews
            selectors = [
                {'tag': 'span', 'attrs': {'data-hook': 'review-body'}},
                {'tag': 'div', 'attrs': {'class': 'a-row a-spacing-small review-data'}},
                {'tag': 'span', 'attrs': {'class': 'a-size-base review-text review-text-content'}},
                {'tag': 'div', 'attrs': {'data-qa': 'review-text'}},
                {'tag': 'span', 'attrs': {'class': 'review-text-content'}}
            ]

            review_data = []

            # Attempt to find reviews using the list of selectors
            for selector in selectors:
                reviews = soup.find_all(selector['tag'], selector['attrs'])
                if reviews:
                    for review in reviews:
                        review_text = review.get_text(strip=True)
                        if review_text:  # Ensure the review isn't empty
                            sentiment = classify_sentiment(review_text)  # Replace with your sentiment analysis
                            review_data.append({"review": review_text, "sentiment": sentiment})
                    break  # Exit loop if reviews are found

            if not review_data:
                st.warning("Could not find reviews. Ensure the product has reviews and check the URL or review section selector.")
                return []

            return review_data

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return []

        finally:
            driver.quit()  # Ensure the driver is closed
        
    
      
    # Function to get YouTube comments using API
    def get_youtube_comments(video_id):
        api_key = 'AIzaSyAR3SZ0pdcqv4Zk2xQA7-HnMUk3prJBGSo'
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        try:
            request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=10)
            response = request.execute()
            
            comments = []
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                sentiment = classify_sentiment(comment)
                comments.append({"comment": comment, "sentiment": sentiment})
            return comments

        except Exception as e:
            st.error(f"Error fetching YouTube comments: {e}")
            return []

    # Download NLTK VADER lexicon
    nltk.download('vader_lexicon')

    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()


    # Helper function to automatically find potential review classes
    def auto_detect_reviews(soup):
        # Look for divs, spans, paragraphs or any elements with a large amount of text that could be reviews
        review_elements = soup.find_all(['div', 'span', 'p', 'li'], text=True)
        
        # Filter review-like content: length of text > 30 chars and no scripts/styles
        reviews = []
        for element in review_elements:
            review_text = element.get_text(strip=True)
            if len(review_text) > 30 and element.name not in ['script', 'style']:  # Ignoring small, irrelevant text
                reviews.append(review_text)

        return reviews

    # Function to scrape reviews from a general URL with pagination
    def scrape_reviews(url, max_reviews=20):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        review_data = []
        page_number = 1

        while len(review_data) < max_reviews:
            try:
                # Modify the URL to account for pagination
                paginated_url = f"{url}?page={page_number}"
                response = requests.get(paginated_url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                # Automatically detect review sections
                reviews = auto_detect_reviews(soup)

                # If no reviews found on this page, break the loop
                if not reviews:
                    break

                for review_text in reviews:
                    sentiment = classify_sentiment(review_text)
                    review_data.append({"review": review_text, "sentiment": sentiment})

                page_number += 1  # Go to the next page

            except requests.RequestException as e:
                st.error(f"HTTP request failed: {e}")
                break
            except Exception as e:
                st.error(f"An error occurred: {e}")
                break

        if not review_data:
            st.warning("Could not find reviews. The page might have a different structure.")
        else:
            return review_data[:max_reviews]  # Return only up to max_reviews

    # Test function for Meesho reviews
    def scrape_meesho_reviews(url):
        return scrape_reviews(url, max_reviews=20)

    # Test function for Myntra reviews
    def scrape_myntra_reviews(url):
        return scrape_reviews(url, max_reviews=20)

    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')

    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    def scrape_flipkart_reviews(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Try to find any div or span that contains reviews based on text patterns
            potential_reviews = soup.find_all(['div', 'span'], text=True)

            review_data = []
            for review_block in potential_reviews:
                review_text = review_block.get_text(strip=True)
                # Look for text that matches review patterns (e.g., non-empty, substantial content)
                if len(review_text) > 30:  # Set a minimum length to filter out irrelevant text
                    sentiment = classify_sentiment(review_text)
                    review_data.append({"review": review_text, "sentiment": sentiment})

            if not review_data:
                st.warning("Could not find reviews. Try checking the class or HTML structure.")
                return []

            return review_data

        except requests.RequestException as e:
            st.error(f"HTTP request failed: {e}")
            return []
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return
     



    
    # Sidebar Menu
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Platform Selection", "About page", "Technical Details"])

    # Platform Selection
    if options == "Platform Selection":
        platform = st.sidebar.selectbox("Choose a platform", ["YouTube", "Amazon", "Flipkart", "Meesho", "Myntra"])
        # YouTube Analysis
        if platform == "YouTube":
            st.header("YouTube Sentiment Analysis")
            video_id = st.text_input("Enter YouTube video ID")
            
            if st.button("Analyze Comments"):
                if video_id:
                    with st.spinner('Fetching comments...'):
                        comments_data = get_youtube_comments(video_id)
                        df = pd.DataFrame(comments_data)
                        if df.empty:
                            st.error("No comments found or there was an issue with fetching.")
                        else:
                            st.write(df)
                            plot_bar_chart(df)
                            plot_pie_chart(df)
                            plot_scatter_plot(df)
                            plot_histogram(df)
                else:
                    st.warning("Please enter a video ID!")

        # Amazon Analysis
        elif platform == "Amazon":
            st.header("Amazon Review Sentiment Analysis")
            url = st.text_input("Enter Amazon product URL")
            
            if st.button("Analyze Reviews"):
                if url:
                    with st.spinner('Scraping reviews...'):
                        review_data = scrape_amazon_reviews(url)
                        if not review_data:
                            st.error("No reviews found or there was an issue with scraping.")
                        else:
                            df = pd.DataFrame(review_data)
                            st.write(df)
                            if 'sentiment' in df.columns:
                                plot_bar_chart(df)
                                plot_pie_chart(df)
                                plot_scatter_plot(df)
                                plot_histogram(df)
                            else:
                                st.error("The 'sentiment' column is missing from the review data.")
                else:
                    st.warning("Please enter a valid product URL!")

        # Flipkart Analysis
        elif platform == "Flipkart":
            st.header("Flipkart Review Sentiment Analysis")
            url = st.text_input("Enter Flipkart product URL")
            
            if st.button("Analyze Reviews"):
                if url:
                    with st.spinner('Scraping reviews...'):
                        review_data = scrape_flipkart_reviews(url)
                        if not review_data:
                            st.error("No reviews found or there was an issue with scraping.")
                        else:
                            df = pd.DataFrame(review_data)
                            st.write(df)
                            if 'sentiment' in df.columns:
                                plot_bar_chart(df)
                                plot_pie_chart(df)
                                plot_scatter_plot(df)
                                plot_histogram(df)
                            else:
                                st.error("The 'sentiment' column is missing from the review data.")
                else:
                    st.warning("Please enter a valid product URL!")

        elif platform == "Meesho":
            st.header("Meesho Review Sentiment Analysis")
            url = st.text_input("Enter Meesho product URL")

            if st.button("Analyze Reviews"):
                if url:
                    with st.spinner('Scraping reviews...'):
                        review_data = scrape_meesho_reviews(url)
                        if not review_data:
                            st.error("No reviews found or there was an issue with scraping.")
                        else:
                            df = pd.DataFrame(review_data)
                            st.write(df)
                            if 'sentiment' in df.columns:
                                plot_bar_chart(df)
                                plot_pie_chart(df)
                                plot_scatter_plot(df)
                                plot_histogram(df)
                            else:
                                st.error("The 'sentiment' column is missing from the review data.")
                else:
                    st.warning("Please enter a valid product URL!")

        # Myntra Analysis
        elif platform == "Myntra":
            st.header("Myntra Review Sentiment Analysis")
            url = st.text_input("Enter Myntra product URL")

            if st.button("Analyze Reviews"):
                if url:
                    with st.spinner('Scraping reviews...'):
                        review_data = scrape_myntra_reviews(url)
                        if not review_data:
                            st.error("No reviews found or there was an issue with scraping.")
                        else:
                            df = pd.DataFrame(review_data)
                            st.write(df)
                            if 'sentiment' in df.columns:
                                plot_bar_chart(df)
                                plot_pie_chart(df)
                                plot_scatter_plot(df)
                                plot_histogram(df)
                            else:
                                st.error("The 'sentiment' column is missing from the review data.")
                else:
                    st.warning("Please enter a valid product URL!")
    # About Page
    elif options == "About page":
        st.header("About Us – Feedback Fusion")
        st.write("""At Feedback Fusion, we specialize in transforming raw feedback from multiple sources into actionable insights. Our advanced sentiment analysis platform is designed to analyze customer opinions across various channels, including social media, e-commerce reviews, and video platforms. Whether it’s Twitter, Instagram, YouTube, Amazon, or popular shopping platforms like Flipkart, Meesho, Myntra, and Ajio, our tool provides a unified, comprehensive view of public sentiment.
        We empower businesses and individuals to understand customer behavior, identify trends, and make data-driven decisions with precision. By categorizing feedback into positive, negative, or neutral sentiments, we help our users stay ahead of the curve, enhancing their customer engagement and brand reputation.
        At Feedback Fusion, we believe that every opinion counts. Our mission is to give you the power to interpret those opinions and use them to fuel your growth and success.
        """)

    # Technical Details Page
    elif options == "Technical Details":
        st.header("Technical Details")
        st.markdown("""
        - **Web Application Framework:**
        - Streamlit: Used to create the interactive web interface.

        - **Data Handling and Analysis:**
        - pandas: For data manipulation and storage in DataFrames.
        - numpy: Used for numerical computations and handling arrays, especially for mathematical operations during analysis.
        - nltk (Natural Language Toolkit): Specifically, SentimentIntensityAnalyzer for sentiment analysis.
        - sklearn: Used for feature extraction (TfidfVectorizer) and machine learning models (Logistic Regression).

        - **Web Scraping:**
        - requests: For sending HTTP requests and fetching web pages.
        - BeautifulSoup (from bs4):To parse HTML content and extract data from websites.

        - **Visualization:**
        - matplotlib.pyplot: To create and display pie charts, scatter plots, and histograms.
        - plotly: For interactive plots and advanced visualizations, such as creating charts with plotly.express and plotly.graph_objects.
        - seaborn: For advanced statistical data visualizations (e.g., scatter and distribution plots).

        - **External APIs:**
        - googleapiclient.discovery: To interact with the YouTube Data API v3 and fetch comments for sentiment analysis.

        - **Other Libraries:**
        - nltk.download('vader_lexicon'): Downloads the VADER lexicon for performing sentiment analysis.
        - scipy: Specifically, scipy.stats for functions like gaussian_kde to create density plots and other statistical analysis.

        """)


    # Footer
    st.sidebar.write("© 2024 Feedback Fusion")
    
    
    
