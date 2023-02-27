ChatGPT took the world by storm and reached over a [100 Million unique users](https://www.reuters.com/technology/chatgpt-sets-record-fastest-growing-user-base-analyst-note-2023-02-01/) since its release, 
which makes it the fastest application ever to hit this milestone and everyone is talking about it. 
If even your non-tech-savvy grandpa asks if you heard of a new, disrupting AI, the chances are that such a technology is overhyped. 
But in this case I‘m not sure if it is overhyped or if it will significantly alter how we search, work and entertain ourselves.


As a data scientist searching for code snippets to improve productivity is part of the job. 
But for more complicated or very specific tasks a google search is not enough and you have to describe your problem to others. 
StackOverflow is exactly filling that void by providing a public platform. 
The disadvantage is that somebody with the appropriate knowledge has to take the time to read your question and answer, 
hopefully in a timely manner. Additionally, you might not want to publish your exact data or problem to a public forum.


ChatGPT could solve these limitations if it is able to answer correctly. 


At the beginning of every new project a data scientist has to become acquainted with the data and bring it to a workable form. 
This can be time consuming and theoretically an AI could much faster scan through the data to find anomalies and propose the suitable preprocessing steps.
So let’s generate some data and test ChatGPT.


{% highlight ruby %}def create_dataset(size):
    """Creating a dataset of Cars"""
    df=pd.DataFrame()
    #creating a Date column with random timestamps using Unix timestamp (1676851200== '2023-02-20 00:00:00')
    # from that timestamp up to 1 year in increments of 1 hour and then converting the timestamp to a string
    df['Date'] = np.random.randint(0, 24*365, size)*60*60 + 1676851200
    df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.strftime('%Y-%m-%d %X')
    
    df['Category'] = np.random.choice(['Gas','Hybrid','Electric'], size)
    df['Year'] = np.random.randint(2010, 2024, size)
    df['Model'] = np.char.add(np.random.choice(['BMW','VW', 'Mercedes'], size) , ['-' + str(i) for i in np.random.randint(1, 1000, size)])
    df['Cylinders'] = np.random.randint(2, 17, size)
    df['Accident'] = np.random.choice(['yes','no'], size)
    df['Score'] = np.random.uniform(1, 0, size)
  
    # Electric cars dont have cylinders and cylinders is set to NaN
    df.loc[df['Category']=='Electric','Cylinders']=np.nan
    return df

{% endhighlight %}
