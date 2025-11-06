Hello, this is my prototype for the APP competition.
This is how to download and use the prototype.

- Linux/macOS:
    - cd /path//to/projectresearchbargithub
    - python3 -m venv .venv && . .venv/bin/activate
    - pip install -r requirements.txt
    - python3 launch.py
- Windows:
    - cd \path\to\projectresearchbargithub
    - py -m venv .venv && .venv\Scripts\activate
    - pip install -r requirements.txt
    - py launch.py.
(Or download the file and ask ChatGPT for help on how to download this stuff and run it)

Download the zip and extract.
If you are on windows run the launch_windows.exe/bat file
If you are on linux run .sh file, launch.sh

In the browser or in its own window, you should see the application open and be presented with a searchbar.
You will need to download a database to perform a search on. 
Go to setttings, and scroll to the bottom, download the papers which takes awhile. Vecotorize them.
Then go to the setting bar above the download settings, the build index button.

You want to run the build index button, then wait awhile till it completes buidling the database.
So download the papers, turn them into vectors using the settings at the bottom. 
Then build an index with them using the option in the settings.

---

Once it's done you can type in key words, latex code, sentences, and other inputs and press search.
Once you search you can run agents and prompt and agent that goes off and reads through the papersb basaed on your prompt.
You can also adjust settings or choose how results are ranked, cosine or bm25.
You can copy exerpts from papers and paste them into the search bar to find them.


---

To use the AI, you will need an API key from open AI.
https://platform.openai.com/api-keys

