<h1>Title Generation using Fine-Tuned GPT Model</h1>

<h2>Title Generation using Fine-Tuned GPT Model</h2>

<p><strong>Title Generation using Fine-Tuned GPT Model</strong> is a project aimed at generating course titles based on a given set of skills using the GPT-2 model.</p>

<h3>Project Overview</h3>

<p>The project focuses on fine-tuning the GPT-2 model to generate relevant and coherent course titles based on input sets of skills. Leveraging a dataset from Kaggle containing Coursera course information, the model is trained to understand the relationship between different skills and their corresponding course titles.</p>

<h3>Contributors</h3>

<ul>
    <li>Faisal Omari, ID: 325616894, University of Haifa</li>
    <li>Saji Assi, ID: 314831207, University of Haifa</li>
</ul>

<h3>Installation</h3>

<ol>
    <li>Clone the repository:</li>
</ol>

<pre><code>git clone https://github.com/faisalomari/title_generation_gpt.git
cd title_generation_gpt
</code></pre>

<ol start="2">
    <li>Install the required packages:</li>
</ol>

<pre><code>pip install -r requirements.txt
</code></pre>

<h3>Usage</h3>

<ol>
    <li><strong>Data Collection and Preprocessing</strong>: Utilize the provided script to preprocess the Coursera course dataset and create training data in the required format.</li>
    <li><strong>Fine-Tuning the GPT Model</strong>: Fine-tune the GPT-2 model using the preprocessed training data to train it on the task of generating course titles.</li>
    <li><strong>Generating Course Titles</strong>: Use the trained GPT-2 model to generate course titles based on input sets of skills.</li>
</ol>

<h3>Results and Discussion</h3>

<p>Refer to the "Results and Discussion" section in the README or documentation for a detailed analysis of the generated course titles and model performance.</p>

<h3>Additional Information</h3>

<ul>
    <li>The training data should be structured as follows:</li>
</ul>

<pre><code>training_data/
├── training_data.txt
</code></pre>

<p>For any questions or issues, feel free to open an issue or contact the contributors.</p>

