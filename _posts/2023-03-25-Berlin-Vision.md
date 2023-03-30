Berlin Vision is hosted on Google Cloud and combines image classification and the use of ChatGPT. 

Berlin Vision utilizes computer vision technology to identify buildings in Berlin. The application is deployed to the cloud, which enables users to easily access it via a web browser. The application provides users with relevant information about the identified building, and the option to request further assistance.


### Functionality
The application's primary functionality is to identify buildings in Berlin from uploaded images. The AI model employed in the application is designed to accurately recognize ten of the most iconic buildings in Berlin. Once the AI model identifies the building, a chatbot will provide the user with three interesting facts about the identified building. The user also has the option to further interact with the chatbot.

### Benefits
While the actual use of Berlin Vision is limited because of the small number of supported buildings, it shows the possibilities when different AI models interact. 
ChatGPT is very successful but could be much more powerful if it interacted with different tools. By providing ChatGPT with an image as input, this app offers a solution to two problems. First, it can identify objects that users may not be able to name. Second, it saves time by eliminating the need to manually describe a scenario.
This can be especially useful for tourists or individuals unfamiliar with the city. There are many other use cases where this would be helpful. For example, it could also be applied for generating a recipe based on an image of ingredients of a fridge, where the AI then also estimate quantities. Or in general making manuals more interactive.

### Image model
The app leverages advanced machine learning technology to identify buildings in Berlin. Specifically, it employs a pretrained model based on Convolutional Neural Networks called EfficientNet. This model is available with TensorFlow and PyTorch, two popular deep learning frameworks.
EfficientNet is trained on the ImageNet project, a large database with over 14 million images of various objects. This makes it well-suited for recognizing buildings in Berlin, given that it has already learned how to distinguish between a wide range of objects.
One of the advantages of EfficientNet is its ability to deliver good performance while using fewer parameters than other models. 
To train the image model transfer learning is used. The weights of the EfficientNet model were frozen and its last fully-connected layer was removed. Then a new Dense layer was added to adapt the model to this specific use case. By using transfer learning, the model  was trained more efficiently and achieve high accuracy with limited data. This was especially beneficial because all data for Berlin Vision had to be scraped from Google Images. In total 150 images of each building were downloaded. After some cleaning of the data and a train-test-split the model was finetuned with 100 images of each class. Typical data augmentation like rotations and zooms were also incorporated into the training pipeline.

### ChatBot
In addition to image recognition, the app also employs a chatbot interface powered by the ChatGPT API. ChatGPT is probably the most popular AI model right now and has recently released an API, making it easy to integrate with web pages.
While ChatGPT is not strictly necessary for providing users with three interesting facts about the identified building (as this information could be hardcoded), it adds a valuable layer of interactivity to the app. The chatbot interface enables users to ask questions about the identified building and receive personalized recommendations based on their interests.
By integrating ChatGPT, the app becomes more than just a tool for identifying buildings in Berlin; it becomes a personalized tour guide that can adapt to the user's needs and interests. If more points of interest were included, Berlin Vision could become a valuable tool for both tourists and locals looking to explore the city.

### Deployment to the cloud
The implementation of the app on Google Cloud involves three main steps.
Firstly, the trained TensorFlow model is saved in a Google Storage bucket. The standard storage is used, and the region is selected to be close to the user for optimal performance.
Secondly, the model is deployed to an endpoint in Vertex AI. The model is connected to the Vertex AI model registry and deployed using pre-built container settings. The Docker container is set up with a specified TensorFlow runtime. It is possible to include a GPU accelerator. However, since inference on a single image is fast, it is not needed. Instead the optimized TensorFlow runtime is enabled, which efficiently utilizes multithreaded host CPUs. The Vertex Python API is then used to obtain predictions from the deployed model.
Finally, the entire app is deployed to App Engine, which is Google's Platform as a Service solution. The app is hosted on the flexible environment, which means that the application instances run within Docker containers on Compute Engine virtual machines. To use the ChatGPT API, the newest version of the openai Python package has to be part of the Docker image, which requires at least Python version 3.7. The code is deployed using the "gcloud app deploy app.yaml" command.
Once the code is deployed, the Docker image is automatically uploaded to Google's Container Registry. The image is then deployed on an App Engine instance, and the app is live and reachable under the project URL. This deployment process ensures that the app is scalable, reliable, and can handle a large number of requests from users.

