__author__ = 'bryanfeeney'

import sys
import operator

InputFile = "/Users/bryanfeeney/Dropbox/user-counts.txt"
TweetTargetCount = 5000000
MinPerUserTweet = 1500
MaxPerUserTweet = 8000
SortTweet = False


def read_counts(filename, min_per_user_tweet = 0, max_per_user_tweet=4000000000):
    '''
    Reads in user counts from the given file
    :param filename: path to the file to read them in
    :return: a dictionary mapping group names to dictionaries,
    in turn mapping user names to counts of tweets.
    '''
    counts = dict()
    with open (filename, 'r') as f:
        for line in f:
            (count, groupname, filename) = line.split("\t")
            username = ".".join(filename.split('.')[:-1]) \
                if '.' in filename \
                else filename
            username = username.strip()

            if not groupname in counts:
                group = dict()
                counts[groupname] = group
            else:
                group = counts[groupname]

            if not username in group:
                group[username] = int(count)
            else:
                group[username] += int(count)

    for user_counts in counts.values():
        odd_users = []
        for user in user_counts.keys():
            tweet_count = user_counts[user]
            if not min_per_user_tweet < tweet_count < max_per_user_tweet:
                odd_users.append(user)

        for user in odd_users:
            del user_counts[user]

    return counts


def select_group_contribs(user_counts, target):
    '''
    Select an equal batch of users from each group until we meet
    the target tweet count.
    :param counts: a dictionary mapping group names to dictionaries,
    in turn mapping user names to counts of tweets.
    :param target: the total size of the dataset
    :return: the list of users to read
    '''
    dataset_total = 0
    group_counts = dict()
    min_count = 3000000000
    for group, dic in user_counts.items():
        group_total = sum(v for v in dic.values())
        group_counts[group] = group_total
        dataset_total += group_total

        if group_total < min_count:
            min_count = group_total

    num_groups    = len (group_counts)
    group_target = target // num_groups

    if group_target < min_count:
        return { group:group_target for group in group_counts.keys() }

    candidates = [g for g in group_counts.keys()]
    group_contribs = {g:0 for g in candidates}
    total_assigned = 0
    while total_assigned < target:
        group_target = 1 + (target - total_assigned) // len(candidates)
        for group in candidates:
            if group_counts[group] < group_target:
                group_contribs[group] += group_counts[group]
                total_assigned        += group_counts[group]
                group_counts[group]    = 0
            else:
                group_contribs[group] += group_target
                total_assigned        += group_target
                group_counts[group]   -= group_target

            if group_counts[group] == 0:
                candidates.remove(group)

    return group_contribs


def select_users(user_counts, group_contribs):
    '''
    Given the contributions from each group, select the users which
    we need to include to hit our target tweet count.
    :param user_counts:
    :param group_contribs:
    :return: the list of users to process
    '''
    selected_users = set()
    for group, contrib in group_contribs.items():
        acc = 0
        sorted_users = sorted(user_counts[group].items(), key=operator.itemgetter(1)) \
            if SortTweet else [(k,v) for k,v in user_counts[group].items()]

        u = []
        while acc < contrib and len(sorted_users) > 0:
            acc += sorted_users[-1][1]
            selected_users.add (sorted_users[-1][0])
            u.append(sorted_users[-1][0])
            del sorted_users[-1]

        print ("%25s  ->   %2d   -> %s" % (group, len(u), str(u)))

    return selected_users




def run(args):
    user_counts = read_counts(InputFile, MinPerUserTweet, MaxPerUserTweet)
    group_contribs = select_group_contribs(user_counts, TweetTargetCount)
    users = select_users(user_counts, group_contribs)
    print (str(len(users)))
    print ("\n".join(users))

if __name__ == '__main__':
    run(args=sys.argv[1:])