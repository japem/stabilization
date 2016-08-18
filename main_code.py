#Jonah Pemstein
#Cronbach's alpha to measure reliability of different stats
#Data sources: 
#   Baseball Info Solutions (batted balls, stolen bases)
#   Pitchf/x (pitches, at bats, plate appearances, pickoffs, left on base)
#   StatCast (statcast stats)
#   Retrosheet (fielding, shutdowns/meltdowns)

#Import modules
import pandas as pd
import numpy as np
import random
import timeit
import math
import os
print("Modules all imported")

#Start time
stime = timeit.default_timer() #start time

#Default file paths
ip = r"/Users/japemstein/Documents/FanGraphs/Stabilization Python/Data/V3/" #Input files
op = r"/Users/japemstein/Documents/FanGraphs/Stabilization Python/Output/V5/" #Output files

#Define functions
def time(start, msg="   "):
    curtime = timeit.default_timer()
    tottime = curtime-start
    hours = math.floor(tottime/3600)
    minutes = math.floor(tottime/60)-hours*60
    seconds = tottime % 60
    if seconds < 10:
        seconds = "0"+str(round(seconds,1))
    else:
        seconds = str(round(seconds,1))
    if minutes < 10:
        minutes = "0"+str(minutes)
    if hours < 10:
        hours = "0"+str(hours)
    print(msg, "Time elapsed: "+str(hours)+":"+str(minutes)+":"+str(seconds))

def alpha(prepped):
    #Calculate Cronbach's alpha       
    stdx = np.std(prepped.sum())
    varx = stdx*stdx #The variance of all total scores
    fpv = prepped.transpose()
    stdy = fpv.std()
    vary = stdy*stdy #The variances of every player-year's scores
    k = prepped.shape[0] #Number of "test items", in this case balls in play 
    kterm = k/(k-1)
    sum_vary = np.sum(vary) #The sum of all variances of total scores
    varterm = 1-(sum_vary/varx)
    return(kterm * varterm)

def calculate(statlist, data, playerdata, playeridtype, yearcolumn, denom_name, yearrange, playertype, path, maxdenom, increment, extradenom=[]):
    #Create dictionary with every increment of denominator desired
    statnum=[]
    for i in range(1, int(maxdenom/increment)):
        statnum.append(i*increment)
    statnum.extend(extradenom)
    statnum.sort()
    stat_dict = {denom_name:statnum}

    alpha_df, mean_df, sd_df, count_df = pd.DataFrame(stat_dict),pd.DataFrame(stat_dict),pd.DataFrame(stat_dict),pd.DataFrame(stat_dict)  #create dataframes with every increment of denominator desired  

    playerlist = pd.Series.tolist(playerdata[playeridtype]) #make a list of all player IDs

    #Iterate through different statistics
    for stat in statlist:
        alpha_list, mean_list, sd_list, count_list = [],[],[],[] #clear list of alphas, means, standard deviations, and sample sizes
        nums_dict = {} #Create empty dictionary
        for i in playerlist: #Populate dictionary with batter numbers for the given statistic
            for y in yearrange:
                nums = pd.Series.tolist(data[(data[playeridtype] == i) & (data[yearcolumn] == y)][stat])
                nums_dict[str(i)+str(y)] = nums

      #Iterate through different numbers of events, each time creating a dataframe that alpha can be calculated from
        for samplesize in stat_dict[denom_name]:
            #Create empty dataframe
            x = {}
            prepped = pd.DataFrame(x)
            #Fill that dataframe with a random sample of events
            for i in nums_dict:
                if len(nums_dict[i]) >= samplesize:
                    prepped[str(i)] = random.sample(nums_dict[i], samplesize) #Add the random sample to the prepped dataframe that will be used to calculate alpha 
         
            if prepped.shape[1] >= 5: #If there are at least five players with enough events, add alpha to the list of alphas for that stat (and mean, standard deviation, and count)
                a = alpha(prepped)
                alpha_list.append(a)
                m = np.mean(prepped.mean())
                mean_list.append(m)
                s = np.std(prepped.mean())
                sd_list.append(s)
                n = prepped.shape[1]
                count_list.append(n)
            else: #There aren't enough batters to calculate alpha
                break #stop calculating alpha for this stat and move on to the next stat
            
        #Add that list of alphas for that stat to the dataframe containing alpha for all stats
        alpha_df, mean_df, sd_df, count_df = alpha_df.loc[:len(alpha_list)-1], mean_df.loc[:len(mean_list)-1], sd_df.loc[:len(sd_list)-1], count_df.loc[:len(count_list)-1]
        alpha_df[stat], mean_df[stat], sd_df[stat], count_df[stat] = (alpha_list, mean_list, sd_list, count_list)
        time(stime,msg="Completed "+stat+" for "+path+"/"+playertype+".")    
    dir = op+path+"/"+playertype+"/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    alpha_df.to_csv(dir+"alpha.csv",index=False)
    mean_df.to_csv(dir+"mean.csv",index=False)
    sd_df.to_csv(dir+"sd.csv",index=False)
    count_df.to_csv(dir+"count.csv",index=False)
    print("Completed", path,"for",playertype.lower())

print("Functions all defined")

#####################################################
#Calculate alpha for all these different types of stats!

#BATTED BALLS
bip = pd.read_csv(ip+"BIP.csv") #BIP data between 2013 and 2015
pitchers = pd.read_csv(ip+"fgid_pitcher.csv") #List of pitchers who pitched between 2013 and 2015
batters = pd.read_csv(ip+"fgid_batter.csv") #List of batters who played between 2013 and 2015
print("Data loaded for batted balls")
calculate(["GB", "FB", "LD", "IFFB", "H", "wOBA", "1B", "2B", "3B", "HR", "Soft", "Med", "Hard"], bip, pitchers, "pitcherid", "Year", "BIP", range(2013, 2016), "Pitchers", "BIP/All", 600, 5, [2, 3, 4]) #IFFB% is popups per batted ball
calculate(["GB", "FB", "LD", "IFFB", "H", "wOBA", "1B", "2B", "3B", "HR", "Soft", "Med", "Hard"], bip[bip["Position"] != "P"], batters, "batterid", "Year", "BIP", range(2013, 2016), "Batters", "BIP/All", 550, 5, [2, 3, 4])

#FLY BALLS
fb = bip[bip["FB"]==1]
calculate(["H","wOBA","HR","IFFB"], fb, pitchers, "pitcherid", "Year", "FB", range(2013, 2016), "Pitchers", "BIP/FB", 300, 5, [2, 3, 4]) #IFFB% is popups per fly ball
calculate(["H","wOBA","HR","IFFB"], fb[fb["Position"] != "P"], batters, "batterid", "Year", "FB", range(2013, 2016), "Batters", "BIP/FB", 300, 5, [2, 3, 4]) #IFFB% is popups per fly ball

#GROUND BALLS
gb = bip[bip["GB"]==1]
calculate(["H","wOBA"], gb, pitchers, "pitcherid", "Year", "GB", range(2013, 2016), "Pitchers", "BIP/GB", 300, 5, [2, 3, 4])
calculate(["H","wOBA"], gb[gb["Position"] != "P"], batters, "batterid", "Year", "GB", range(2013, 2016), "Batters", "BIP/GB", 300, 5, [2, 3, 4])

#PLATE APPEARANCES
pa = pd.read_csv(ip+"PA.csv") #Plate appearance data between 2013 and 2015
pitchers = pd.read_csv(ip+"pfx_pitcher.csv") #List of pitchers who pitched between 2013 and 2015
batters = pd.read_csv(ip+"pfx_batter.csv") #List of batters who pitched between 2013 and 2015
print("Data loaded for plate appearances")
calculate(["K", "BB", "HBP", "OBP", "wOBA", "Kcalled", "Kswinging", "R", "ER", "FIP", "xFIP"], pa, pitchers, "pitcher", "Year", "PA", range(2013, 2016), "Pitchers", "PA", 800, 5)
calculate(["K", "BB", "HBP", "OBP", "wOBA", "Kcalled", "Kswinging", "RBI"], pa[pa["Position"] != "P"], batters, "batter", "Year", "PA", range(2013, 2016), "Batters", "PA", 700, 5)

#PITCHES
pitch = pd.read_csv(ip+"Pitch.csv") #pitch by pitch data between 2013 and 2015
pitchers = pd.read_csv(ip+"pfx_pitcher.csv") #List of pitchers who pitched between 2013 and 2015
batters = pd.read_csv(ip+"pfx_batter.csv") #List of batters who saw pitches between 2013 and 2015
print("Data loaded for pitch by pitch")
calculate(["SwStr", "Swing", "Zone","Strike","FB", "BB", "CH", "x0", "z0"], pitch, pitchers, "pitcher", "Year", "Pitches", range(2013, 2016), "Pitchers", "Pitch/All", 3500, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])
calculate(["SwStr", "Swing", "Zone"], pitch[pitch["Position"] != "P"], batters, "batter", "Year", "Pitches", range(2013, 2016), "Batters", "Pitch/All", 3000, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])

#FASTBALLS
fb = pitch[pitch["pitch_type"] == "FF"]
calculate(["start_speed","spin_rate","SwStr", "Swing", "Zone", "RV"], fb, pitchers, "pitcher", "Year", "Pitches", range(2013, 2016), "Pitchers", "Pitch/Fastballs", 2000, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])
calculate(["SwStr", "Swing", "Zone", "RV"], fb[fb["Position"] != "P"], batters, "batter", "Year", "Pitches", range(2013, 2016), "Batters", "Pitch/Fastballs", 1000, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])

#CHANGEUPS
ch = pitch[pitch["pitch_type"] == "CH"]
calculate(["start_speed","spin_rate","SwStr", "Swing", "Zone", "RV"], ch, pitchers, "pitcher", "Year", "Pitches", range(2013, 2016), "Pitchers", "Pitch/Changeups", 800, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])
calculate(["SwStr", "Swing", "Zone", "RV"], ch[ch["Position"] != "P"], batters, "batter", "Year", "Pitches", range(2013, 2016), "Batters", "Pitch/Changeups", 300, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])

#SINKERS
si = pitch[(pitch["pitch_type"] == "FT") | (pitch["pitch_type"] == "SI")]
calculate(["start_speed","spin_rate","SwStr", "Swing", "Zone", "RV"], si, pitchers, "pitcher", "Year", "Pitches", range(2013, 2016), "Pitchers", "Pitch/Sinkers", 1600, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])
calculate(["SwStr", "Swing", "Zone", "RV"], si[si["Position"] != "P"], batters, "batter", "Year", "Pitches", range(2013, 2016), "Batters", "Pitch/Sinkers", 700, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])

#CURVEBALLS
cu = pitch[pitch["pitch_type"] == "CU"]
calculate(["start_speed","spin_rate","SwStr", "Swing", "Zone", "RV", "pfx_x", "pfx_z"], cu, pitchers, "pitcher", "Year", "Pitches", range(2013, 2016), "Pitchers", "Pitch/Curveballs", 800, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])
calculate(["SwStr", "Swing", "Zone", "RV"], cu[cu["Position"] != "P"], batters, "batter", "Year", "Pitches", range(2013, 2016), "Batters", "Pitch/Curveballs", 300, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])

#SLIDERS
sl = pitch[pitch["pitch_type"] == "SL"]
calculate(["start_speed","spin_rate","SwStr", "Swing", "Zone", "RV", "pfx_x", "pfx_z"], sl, pitchers, "pitcher", "Year", "Pitches", range(2013, 2016), "Pitchers", "Pitch/Sliders", 1200, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])
calculate(["SwStr", "Swing", "Zone", "RV"], sl[sl["Position"] != "P"], batters, "batter", "Year", "Pitches", range(2013, 2016), "Batters", "Pitch/Sliders", 500, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])

#IN-ZONE ONLY
zone = pitch[pitch["Zone"] == 1]
calculate(["Swing"], zone, pitchers, "pitcher", "Year", "Pitches", range(2013, 2016), "Pitchers", "Pitch/Zone", 1800, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])
calculate(["Swing"], zone[zone["Position"] != "P"], batters, "batter", "Year", "Pitches", range(2013, 2016), "Batters", "Pitch/Zone", 1400, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])

#OUT-OF-ZONE ONLY
ooz = pitch[pitch["Zone"] == 0]
calculate(["Swing"], ooz, pitchers, "pitcher", "Year", "Pitches", range(2013, 2016), "Pitchers", "Pitch/OOZ", 1800, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])
calculate(["Swing"], ooz[ooz["Position"] != "P"], batters, "batter", "Year", "Pitches", range(2013, 2016), "Batters", "Pitch/OOZ", 1400, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])

#SWINGS ONLY
swing = pitch[pitch["Swing"] == 1]
calculate(["Contact","Foul"], swing, pitchers, "pitcher", "Year", "Pitches", range(2013, 2016), "Pitchers", "Pitch/Swing", 1800, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])
calculate(["Contact","Foul"], swing[swing["Position"] != "P"], batters, "batter", "Year", "Pitches", range(2013, 2016), "Batters", "Pitch/Swing", 1400, 20, [2, 3, 4, 5, 10, 15, 25, 30, 50])

#STATCAST
sc = pd.read_csv(ip+"Statcast.csv") #statcast data from 2016
pitchers = pd.read_csv(ip+"pfx_pitcher.csv") #List of pitchers who pitched in 2016
batters = pd.read_csv(ip+"pfx_batter.csv") #List of batters who saw pitches in 2016
print("Data loaded for Statcast")
calculate(["hit_speed", "hit_distance_sc", "hit_angle"], sc, pitchers, "pitcher", "game_year", "BIP", range(2016, 2017), "Pitchers", "BIP/Statcast", 200, 5, extradenom = [2,3,4,6,7,8,9])
calculate(["hit_speed", "hit_distance_sc", "hit_angle"], sc[sc["Position"] != "P"], batters, "batter", "game_year", "BIP", range(2016, 2017), "Batters", "BIP/Statcast", 200, 5, extradenom = [2,3,4,6,7,8,9])

#AT BATS
ab = pd.read_csv(ip+"AB.csv") #At bat data between 2013 and 2015
pitchers = pd.read_csv(ip+"pfx_pitcher.csv") #List of pitchers between 2013 and 2015
batters = pd.read_csv(ip+"pfx_batter.csv") #List of batters who had at bats between 2013 and 2015
print("Data loaded for at bats")
calculate(["AVG", "SLG", "ISO"], ab, pitchers, "pitcher", "Year", "AB", range(2013, 2016), "Pitchers", "AtBats", 600, 5, extradenom = [2,3,4])
calculate(["AVG", "SLG", "ISO"], ab[ab["Position"] != "P"], batters, "batter", "Year", "AB", range(2013, 2016), "Batters", "AtBats", 600, 5, extradenom = [2,3,4])

#FIELDING
fld = pd.read_csv(ip+"Fielding.csv") #Plate appearance data between 2013 and 2015
fielders = pd.read_csv(ip+"fgid_fielder.csv") #List of batters who pitched between 2013 and 2015
print("Data loaded for fielding")
calculate(["FieldingPct"], fld, fielders, "FielderId", "Year", "TC", range(2013, 2016), "Fielders", "Fielding", 300, 5, extradenom = [2,3,4])

#STOLEN BASES
sb = pd.read_csv(ip+"SB.csv") #Stolen base attempts between 2013 and 2015
runners = pd.read_csv(ip+"fgid_runner.csv") #List of players who attempted a steal between 2013 and 2015
pitchers = pd.read_csv(ip+"fgid_pitcher.csv") #List of pitchers who were stolen on (attempted) between 2013 and 2015
catchers = pd.read_csv(ip+"fgid_catcher.csv") #List of catchers who were stolen on (attempted) between 2013 and 2015
print("Data loaded for stolen bases")
calculate(["SB"], sb, runners, "playerid", "Year", "SBA", range(2013,2016), "Baserunners", "StolenBases", 50, 2)
calculate(["SB"], sb, pitchers, "pitcherid", "Year", "SBA", range(2013,2016), "Pitchers", "StolenBases", 40, 2)
calculate(["SB"], sb, catchers, "catcherid", "Year", "SBA", range(2013,2016), "Catchers", "StolenBases", 90, 2)

#PICKOFFS
pickoff = pd.read_csv(ip+"Pickoffs.csv") #Pickoff attemtps between 2013 and 2015
pitchers = pd.read_csv(ip+"pfx_pitcher.csv") #List of pitchers
print("Data loaded for pickoffs")
calculate(["Pickoff"], pickoff, pitchers, "pitcher", "Year", "POA", range(2013,2016), "Pitchers", "Pickoffs", 180, 5, extradenom = [2,3,4])

#SHUTDOWNS/MELTDOWNS
sdmd = pd.read_csv(ip+"SDMD.csv") #Shutdowns and meltdowns per relief appearance between 2013 and 2015
pitchers = pd.read_csv(ip+"fgid_pitcher.csv") #List of pitchers
print("Data loaded for shutdowns/meltdowns")
calculate(["SD","MD"], sdmd, pitchers, "pitcherid", "Year", "RAPP", range(2013,2016), "Pitchers", "ShutdonwMeltdown", 76, 2)

#LEFT-ON-BASE%
lob = pd.read_csv(ip+"LOB.csv") #Baserunners and whether or not they score between 2013 and 2015
pitchers = pd.read_csv(ip+"pfx_pitcher.csv") #List of pitchers
print("Data loaded for left-on-base%")
calculate(["LOB"], lob, pitchers, "pitcher", "Year", "Baserunners", range(2013,2016), "Pitchers", "LeftOnBase", 280, 5, extradenom = [2,3,4])

#END TIME
time(stime, msg="Everything completed.")