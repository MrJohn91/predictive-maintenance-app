# predictive-maintenance-app

## Summary
This project is a Streamlit-based web application designed for predictive maintenance of Automotive engines. The app uses machine learning to predict when a vehicle engine is likely to need maintenance based on sensor data. For this project, we used test data to demonstrate how the model works in predicting maintenance needs. Additionally, Langchain and an AI-powered chatbot were integrated into the app to provide explanations for the predictions, offering deeper insights into why maintenance is recommended.

## Languages and Libraries Used
- **Python**: Core programming language for data processing and model development
- **Streamlit**: For building an interactive and user-friendly web app
- **Scikit-learn**: Utilized for developing the predictive machine learning model
- **Pandas**: For data manipulation
- **Matplotlib and Seaborn**: For creating visualizations of engine conditions and model insights
- **Langchain**: To facilitate the integration of generative AI and manage prompt flows
- **OpenAI API**: For integrating a chatbot that explains the sensor data and model predictions
- **Joblib**: For saving and loading the trained machine learning models
- **dotenv**: For securely managing API keys and environment variables

## Key Learnings
- Developed proficiency in building interactive web apps using Streamlit and integrating them with machine learning models.
- Learned to utilize Langchain to streamline AI chatbot interactions and prompt management, enabling smooth integration with predictive models.
- Practiced simulating real-world data analysis by using test data to demonstrate how the predictive maintenance model works.
- Improved my ability to communicate technical results by using visualizations and a chatbot that translates complex predictions into easily understandable insights.

## Challenges Overcame
- **Interpreting test data**: Ensuring the model's predictions based on test data were clear to users required thoughtful design. This was solved by using Langchain to manage chatbot interactions that explain the technical predictions.
- **Model performance**: Balancing the model's false positives and false negatives was challenging but critical for achieving accurate predictions. Adjustments and parameter tuning helped make the predictions more reliable.
- **Simplifying explanations**: With the help of Langchain and the OpenAI API, I was able to provide non-technical users with understandable explanations for maintenance predictions, which improved the user experience.

## Additional Reflections
Currently, the app works with test data to showcase its predictive capabilities. Moving forward, integrating real-time data from vehicle sensors would be the next step, enabling real-time monitoring and more proactive maintenance actions. Langchain's ability to manage more complex query flows opens up the possibility of expanding the chatbot to handle more advanced diagnostics and insights. Future improvements could include real-time alerts for maintenance actions and expanded reporting capabilities, making this tool even more practical for fleet managers or vehicle maintenance professionals.
