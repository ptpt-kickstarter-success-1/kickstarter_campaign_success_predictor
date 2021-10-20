import pandas as pd
from datetime import datetime


def wrangle(df):
    # View all rows of the dataframe
    pd.set_option('display.max_columns', None)
    print('df shape:', df.shape)

    # Drop null columns, duplicate columns, columns with 1 value, leaky columns
    df = df.drop(columns=['friends', 'is_backing', 'is_starred', 'permissions',
                          'country_displayable_name', 'currency_symbol', 'usd_type',
                          'category', 'profile', 'creator', 'location', 'slug',
                          'last_update_published_at', 'unread_messages_count',
                          'unseen_activity_count', 'converted_pledged_amount'])
    print('df shape: ', df.shape)

    # Convert time integer to datetime- changed to pd function from apply 
    df['created_at'] = pd.to_datetime(df['created_at'], unit='s')
    df['deadline'] = pd.to_datetime(df['deadline'], unit='s')
    df['launched_at'] = pd.to_datetime(df['launched_at'], unit='s')
    df['state_changed_at'] = pd.to_datetime(df['state_changed_at'], unit='s')

    # create time difference columns
    df['pre_campaign'] = df['launched_at'] - df['created_at']
    df['planned_campaign'] = df['deadline'] - df['launched_at']
    df['actual_campaign'] = df['state_changed_at'] - df['launched_at']
    df['post_campaign'] = df['state_changed_at'] - df['deadline']

    # Extract components from created_at, then drop the original column
    df['year_created'] = df['created_at'].dt.year
    df['month_created'] = df['created_at'].dt.month
    df['day_created'] = df['created_at'].dt.day
    df['weekday_created'] = df['created_at'].dt.day_name()
    df = df.drop(columns='created_at')

    # Extract components from deadline, then drop the original column
    df['year_deadline'] = df['deadline'].dt.year
    df['month_deadline'] = df['deadline'].dt.month
    df['day_deadline'] = df['deadline'].dt.day
    df['weekday_deadline'] = df['deadline'].dt.day_name()
    df = df.drop(columns='deadline')

    # Extract components from launched_at, then drop the original column
    df['year_launched'] = df['launched_at'].dt.year
    df['month_launched'] = df['launched_at'].dt.month
    df['day_launched'] = df['launched_at'].dt.day
    df['weekday_launched'] = df['launched_at'].dt.day_name()
    df = df.drop(columns='launched_at')

    # Extract components from state_changed_at, then drop the original column
    df['year_state_changed'] = df['state_changed_at'].dt.year
    df['month_state_changed'] = df['state_changed_at'].dt.month
    df['day_state_changed'] = df['state_changed_at'].dt.day
    df['weekday_state_changed'] = df['state_changed_at'].dt.day_name()
    df = df.drop(columns='state_changed_at')

    # Engineer other metrics
    df.set_index('id', inplace=True)
    df['average_pledge_amount'] = df['backers_count'] / df['usd_pledged']
    df['average_pledge_amount'] = df['average_pledge_amount'].fillna(0)
    df['percent_pledged'] = (df['pledged'] / df['goal']) * 100

    return df