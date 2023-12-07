import pandas as pd
import math

def get_sim_df (fn):

    print('loading... %s' % fn)
    df = pd.read_csv(fn).apply(pd.to_numeric,errors='ignore')
    
    #print df.to_string();print '------'
    
    group = df.groupby(['timestamp'])
    return group

def get_sim_df_trade (fn):

    print('loading... %s' % fn)
    df = pd.read_csv(fn).apply(pd.to_numeric,errors='ignore')
    
    group = df.groupby(['timestamp'])
    return group

def mean_reversion(gr_bid_level, gr_ask_level):
    bid_mean = gr_bid_level['price'].mean()
    ask_mean = gr_ask_level['price'].mean()
    mean_diff = ask_mean - bid_mean
    return mean_diff

def faster_calc_indicators(raw_fn):
    
    start_time = timeit.default_timer()

    # FROM CSV FILES (DAILY)
    group_o = get_sim_df(raw_book_csv(raw_fn, ('%s-%s-%s' % (_tag, exchange, currency))))
    group_t = get_sim_df_trade(raw_trade_csv(raw_fn, ('%s-%s-%s' % (_tag, exchange, currency)))) #fix for book-1 regression
    
    delay = timeit.default_timer() - start_time
    print('df loading delay: %.2fs' % delay)
     
    level_1 = 2 
    level_2 = 5

    print('param levels', exchange, currency, level_1, level_2)

    #(ratio, level, interval seconds )   
    book_imbalance_params = [(0.2,level_1,1),(0.2,level_2,1)] 
    book_delta_params = [(0.2,level_1,1),(0.2,level_1,5),(0.2,level_1,15), (0.2,level_2,1),(0.2,level_2,5),(0.2,level_2,15)]
    trade_indicator_params = [(0.2,level_1,1),(0.2,level_1,5),(0.2,level_1,15)]

    variables = {}
    _dict = {}
    _dict_indicators = {}

    for p in book_imbalance_params:
        indicator = 'BI'
        _dict.update( {(indicator, p): []} )
        _dict_var = init_indicator_var(indicator, p)
        variables.update({(indicator, p): _dict_var})
        
    for p in book_delta_params:
        indicator = 'BDv1'
        _dict.update( {(indicator, p): []} )
        _dict_var = init_indicator_var(indicator, p)
        variables.update({(indicator, p): _dict_var})
        
        indicator = 'BDv2'
        _dict.update( {(indicator, p): []} )
        _dict_var = init_indicator_var(indicator, p)
        variables.update({(indicator, p): _dict_var})

        indicator = 'BDv3'
        _dict.update( {(indicator, p): []} )
        _dict_var = init_indicator_var(indicator, p)
        variables.update({(indicator, p): _dict_var})

    for p in add_norm_fn(trade_indicator_params):

        indicator = 'TIv1'
        _dict.update( {(indicator, p): []} )
        _dict_var = init_indicator_var(indicator, p)
        variables.update({(indicator, p): _dict_var})
 
        indicator = 'TIv2'
        _dict.update( {(indicator, p): []} )
        _dict_var = init_indicator_var(indicator, p)
        variables.update({(indicator, p): _dict_var})

    _timestamp = []
    _mid_price = []

    seq = 0
    print('total groups:', len(group_o.size().index), len(group_t.size().index))
    
    #main part
    for (gr_o, gr_t) in itertools.izip (group_o, group_t):
        
        if gr_o is None or gr_t is None:
            print('Warning: group is empty')
            continue
        
        if (wrong_trade_time_diff(gr_t[1])):
            #print 'Warning: trade_time is big'
            continue

        timestamp = (gr_o[1].iloc[0])['timestamp']
        
        if banded:
            gr_o = agg_order_book(gr_o[1], timestamp)
            gr_o = gr_o.reset_index(); del gr_o['index']
        else:
            gr_o = gr_o[1]
 
        gr_t = gr_t[1]

        gr_bid_level = gr_o[(gr_o.type == 0)]
        gr_ask_level = gr_o[(gr_o.type == 1)]
        diff = get_diff_count_units(gr_t)

        mid_price, bid, ask, bid_qty, ask_qty = cal_mid_price(gr_bid_level, gr_ask_level, gr_t)

        if bid >= ask:
            seq += 1
            continue

        _timestamp.append (timestamp)
        _mid_price.append (mid_price)
        
        _dict_group = {}
        for (indicator, p) in _dict.keys(): #indicator_fn, param
            level = p[1]
            if level not in _dict_group:
                
                orig_level = level
                level = min (level, len(gr_bid_level), len(gr_ask_level))
                
                _dict_group[level] = (gr_bid_level.head(level), gr_ask_level.head(level))
                
            p1 = () 
            if len(p) == 3:
                p1 = (p[0], level, p[2]) 
            if len(p) == 4:
                p1 = (p[0], level, p[2], p[3]) 
            
            #print indicator

            _i = _l_indicator_fn[indicator](p1, _dict_group[level][0], _dict_group[level][1], diff, variables[(indicator,p)], mid_price)
            _dict[(indicator,p)].append(_i)
        
        for (indicator, p) in _dict.keys(): #indicator_fn, param
            
            col_name = '%s-%s-%s-%s' % (_l_indicator_name[indicator].replace('_','-'),p[0],p[1],p[2])
            if indicator == 'TIv1' or indicator == 'TIv2':
                col_name = '%s-%s-%s-%s-%s' % (_l_indicator_name[indicator].replace('_','-'),p[0],p[1],p[2],p[3])
            
            _dict_indicators[col_name] = _dict[(indicator,p)]

        _dict_indicators['timestamp'] = _timestamp
        _dict_indicators['mid_price'] = _mid_price

        seq += 1
        #print seq,

    fn = indicators_csv (raw_fn)
    df_dict_to_csv (_dict_indicators, fn)

# Feature: calculating 'bookD' using orderbook and trade

# @params

# gr_bid_level: all bid level
# gr_ask_level: all ask level
# diff: summary of trade, refer to get_diff_count_units()
# var: can be empty
# mid: midprice

def live_cal_book_d_v1(param, gr_bid_level, gr_ask_level, diff, var, mid):

    #print gr_bid_level
    #print gr_ask_level

    ratio = param[0]; level = param[1]; interval = param[2]
    #print ('processing... %s %s,level:%s,interval:%s' % (sys._getframe().f_code.co_name,ratio,level,interval)), 

    decay = math.exp(-1.0/interval)
    
    _flag = var['_flag']
    prevBidQty = var['prevBidQty']
    prevAskQty = var['prevAskQty']
    prevBidTop = var['prevBidTop']
    prevAskTop = var['prevAskTop']
    bidSideAdd = var['bidSideAdd']
    bidSideDelete = var['bidSideDelete']
    askSideAdd = var['askSideAdd']
    askSideDelete = var['askSideDelete']
    bidSideTrade = var['bidSideTrade']
    askSideTrade = var['askSideTrade']
    bidSideFlip = var['bidSideFlip']
    askSideFlip = var['askSideFlip']
    bidSideCount = var['bidSideCount']
    askSideCount = var['askSideCount'] 
  
    curBidQty = gr_bid_level['quantity'].sum()
    curAskQty = gr_ask_level['quantity'].sum()
    curBidTop = gr_bid_level.iloc[0].price #what is current bid top?
    curAskTop = gr_ask_level.iloc[0].price

    #curBidQty = gr_r[(gr_r['type']==0)].quantity.sum()
    #curAskQty = gr_r[(gr_r['type']==1)].quantity.sum()
    #curBidTop = gr_r.iloc[0].price #what is current bid top?
    #curAskTop = gr_r.iloc[level].price


    if _flag:
        var['prevBidQty'] = curBidQty
        var['prevAskQty'] = curAskQty
        var['prevBidTop'] = curBidTop
        var['prevAskTop'] = curAskTop
        var['_flag'] = False
        return 0.0
        
    if curBidQty > prevBidQty:
        bidSideAdd += 1
        bidSideCount += 1
    if curBidQty < prevBidQty:
        bidSideDelete += 1
        bidSideCount += 1
    if curAskQty > prevAskQty:
        askSideAdd += 1
        askSideCount += 1
    if curAskQty < prevAskQty:
        askSideDelete += 1
        askSideCount += 1
        
    if curBidTop < prevBidTop:
        bidSideFlip += 1
        bidSideCount += 1
    if curAskTop > prevAskTop:
        askSideFlip += 1
        askSideCount += 1

    
    (_count_1, _count_0, _units_traded_1, _units_traded_0, _price_1, _price_0) = diff

    #_count_1 = (diff[(diff['type']==1)])['count'].reset_index(drop=True).get(0,0)
    #_count_0 = (diff[(diff['type']==0)])['count'].reset_index(drop=True).get(0,0)
    
    bidSideTrade += _count_1
    bidSideCount += _count_1
    
    askSideTrade += _count_0
    askSideCount += _count_0
    

    if bidSideCount == 0:
        bidSideCount = 1
    if askSideCount == 0:
        askSideCount = 1

    bidBookV = (-bidSideDelete + bidSideAdd - bidSideFlip) / (bidSideCount**ratio)
    askBookV = (askSideDelete - askSideAdd + askSideFlip ) / (askSideCount**ratio)
    tradeV = (askSideTrade/askSideCount**ratio) - (bidSideTrade / bidSideCount**ratio)
    bookDIndicator = askBookV + bidBookV + tradeV
        
       
    var['bidSideCount'] = bidSideCount * decay #exponential decay
    var['askSideCount'] = askSideCount * decay
    var['bidSideAdd'] = bidSideAdd * decay
    var['bidSideDelete'] = bidSideDelete * decay
    var['askSideAdd'] = askSideAdd * decay
    var['askSideDelete'] = askSideDelete * decay
    var['bidSideTrade'] = bidSideTrade * decay
    var['askSideTrade'] = askSideTrade * decay
    var['bidSideFlip'] = bidSideFlip * decay
    var['askSideFlip'] = askSideFlip * decay

    var['prevBidQty'] = curBidQty
    var['prevAskQty'] = curAskQty
    var['prevBidTop'] = curBidTop
    var['prevAskTop'] = curAskTop
    #var['df1'] = df1
 
    return bookDIndicator


def get_diff_count_units (diff):
    
    _count_1 = _count_0 = _units_traded_1 = _units_traded_0 = 0
    _price_1 = _price_0 = 0

    diff_len = len (diff)
    if diff_len == 1:
        row = diff.iloc[0]
        if row['type'] == 1:
            _count_1 = row['count']
            _units_traded_1 = row['units_traded']
            _price_1 = row['price']
        else:
            _count_0 = row['count']
            _units_traded_0 = row['units_traded']
            _price_0 = row['price']

        return (_count_1, _count_0, _units_traded_1, _units_traded_0, _price_1, _price_0)

    elif diff_len == 2:
        row_1 = diff.iloc[1]
        row_0 = diff.iloc[0]
        _count_1 = row_1['count']
        _count_0 = row_0['count']

        _units_traded_1 = row_1['units_traded']
        _units_traded_0 = row_0['units_traded']
        
        _price_1 = row_1['price']
        _price_0 = row_0['price']

        return (_count_1, _count_0, _units_traded_1, _units_traded_0, _price_1, _price_0)

# Feature: calculating 'bookI' using orderbook 
# book imbalance

# @params

# gr_bid_level: all bid level
# gr_ask_level: all ask level
# diff: summary of trade, refer to get_diff_count_units()
# var: can be empty
# mid: midprice


def live_cal_book_i_v1(param, gr_bid_level, gr_ask_level, diff, var, mid):
    
    mid_price = mid

    ratio = param[0]; level = param[1]; interval = param[2]
    #print ('processing... %s %s,level:%s,interval:%s' % (sys._getframe().f_code.co_name,ratio,level,interval)), 
    
        
    _flag = var['_flag']
        
    if _flag: #skipping first line
        var['_flag'] = False
        return 0.0

    quant_v_bid = gr_bid_level.quantity**ratio
    price_v_bid = gr_bid_level.price * quant_v_bid

    quant_v_ask = gr_ask_level.quantity**ratio
    price_v_ask = gr_ask_level.price * quant_v_ask
 
    #quant_v_bid = gr_r[(gr_r['type']==0)].quantity**ratio
    #price_v_bid = gr_r[(gr_r['type']==0)].price * quant_v_bid

    #quant_v_ask = gr_r[(gr_r['type']==1)].quantity**ratio
    #price_v_ask = gr_r[(gr_r['type']==1)].price * quant_v_ask
        
    askQty = quant_v_ask.values.sum()
    bidPx = price_v_bid.values.sum()
    bidQty = quant_v_bid.values.sum()
    askPx = price_v_ask.values.sum()
    bid_ask_spread = interval
        
    book_price = 0 #because of warning, divisible by 0
    if bidQty > 0 and askQty > 0:
        book_price = (((askQty*bidPx)/bidQty) + ((bidQty*askPx)/askQty)) / (bidQty+askQty)

        
    indicator_value = (book_price - mid_price) / bid_ask_spread
    #indicator_value = (book_price - mid_price)
    
    return indicator_value

# Feature: calculating midprice using orderbook

# @params

# gr_bid_level: all bid level
# gr_ask_level: all ask level
# group_t: trade data

def cal_mid_price (gr_bid_level, gr_ask_level, group_t):
    
    level = 5 
    mid_type = 'wt'
    #gr_rB = gr_bid_level.head(level)
    #gr_rT = gr_ask_level.head(level)
    
    if len(gr_bid_level) > 0 and len(gr_ask_level) > 0:
        bid_top_price = gr_bid_level.iloc[0].price
        bid_top_level_qty = gr_bid_level.iloc[0].quantity
        ask_top_price = gr_ask_level.iloc[0].price
        ask_top_level_qty = gr_ask_level.iloc[0].quantity
        mid_price = (bid_top_price + ask_top_price) * 0.5 #what is mid price?
    
        if mid_type == 'wt':
            mid_price = ((gr_bid_level.head(level))['price'].mean() + (gr_ask_level.head(level))['price'].mean()) * 0.5
        elif mid_type == 'mkt':
            mid_price = ((bid_top_price*ask_top_level_qty) + (ask_top_price*bid_top_level_qty))/(bid_top_level_qty+ask_top_level_qty)
            mid_price = truncate(mid_price, 1)
        elif mid_type == 'vwap':
            mid_price = (group_t['total'].sum())/(group_t['units_traded'].sum())
            mid_price = truncate(mid_price, 1)
        
        #print mid_type, mid_price

        return (mid_price, bid_top_price, ask_top_price, bid_top_level_qty, ask_top_level_qty)

    else:
        print('Error: serious cal_mid_price')
        return (-1, -1, -2, -1, -1)

def wrong_trade_time_diff(df):
    if len(df) < 2:
        return False  # trade 데이터가 2개 미만이면 문제가 없다고 가정

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_diff = df['timestamp'].diff().dropna().max()

    # max_time_diff() 함수를 호출하여 그 결과를 가져와서 비교
    return time_diff > max_time_diff()


    
def agg_order_book(order_book, timestamp):
    # 여기에 order_book을 처리하는 코드를 작성해주세요.
    # 필요한 경우 timestamp를 활용하여 적절한 처리를 수행하세요.
    
    # 예시로 order_book을 더미 데이터로 변환하는 코드를 작성했습니다.
    # 실제 데이터에 맞게 수정이 필요합니다.
    aggregated_order_book = order_book.copy()
    aggregated_order_book['timestamp'] = timestamp
    return aggregated_order_book
    
def max_time_diff():
    # max_time_diff 함수의 정의
    YOUR_MAX_TIME_DIFF_VALUE = pd.Timedelta(seconds=1)
    return YOUR_MAX_TIME_DIFF_VALUE  # YOUR_MAX_TIME_DIFF_VALUE는 실제로 사용하려는 시간 간격의 최댓값입니다.


    
def main_processing(raw_fn1, raw_fn2):
    # Loading data and grouping
    group_o = get_sim_df(raw_fn1)
    group_t = get_sim_df_trade(raw_fn2)
    
    results = []

    # Initializing variables and parameters
    param_bookD = (0.2, 5, 1)  # Example parameter for 'bookD' feature
    param_bookI = (0.2, 5, 1)  # Example parameter for 'bookI' feature
    var_bookD = {'_flag': True, 'prevBidQty': 0, 'prevAskQty': 0, 'prevBidTop': 0, 'prevAskTop': 0,
                 'bidSideAdd': 0, 'bidSideDelete': 0, 'askSideAdd': 0, 'askSideDelete': 0,
                 'bidSideTrade': 0, 'askSideTrade': 0, 'bidSideFlip': 0, 'askSideFlip': 0,
                 'bidSideCount': 0, 'askSideCount': 0}
    var_bookI = {'_flag': True}

    # Processing each group
    for (gr_o, gr_t) in zip(group_o, group_t):
        if gr_o is None or gr_t is None:
            print('Warning: group is empty')
            continue

        if wrong_trade_time_diff(gr_t[1]):
            continue

        timestamp = (gr_o[1].iloc[0])['timestamp']

        # 'banded' 변수 대신에 'agg_order_book' 함수를 무조건 호출
        gr_o = agg_order_book(gr_o[1], timestamp)
        gr_o = gr_o.reset_index(); del gr_o['index']

        gr_t = gr_t[1]
        
        gr_o['timestamp'] = pd.to_datetime(gr_o['timestamp'])
        gr_t['timestamp'] = pd.to_datetime(gr_t['timestamp'])
        
        gr_o.sort_values('timestamp', inplace=True)
        gr_t.sort_values('timestamp', inplace=True)

        gr_bid_level = gr_o[(gr_o.type == 0)]
        gr_ask_level = gr_o[(gr_o.type == 1)]
        diff = get_diff_count_units(gr_t)

        mid_price, bid, ask, bid_qty, ask_qty = cal_mid_price(gr_bid_level, gr_ask_level, gr_t)

        # Calculating 'bookD' feature
        bookD_value = live_cal_book_d_v1(param_bookD, gr_bid_level, gr_ask_level, diff, var_bookD, mid_price)

        mean_diff = mean_reversion(gr_bid_level, gr_ask_level)
        
        
        
        # Calculating 'bookI' feature
        bookI_value = live_cal_book_i_v1(param_bookI, gr_bid_level, gr_ask_level, diff, var_bookI, mid_price)

        results.append({
            'bookD': bookD_value,
            'bookI': bookI_value,
            'midprice': mid_price,
            'mean_reversion': mean_diff,
            'timestamp': timestamp
        })
        
    # Convert the list of dictionaries to a pandas DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv_path, index=False)
        # Use the calculated features as needed
        
output_csv_path = "C:/Users/82108/Desktop/results_output.csv"

main_processing("C:/Users/82108/Desktop/2023-11-15-upbit-BTC-book.csv", "C:/Users/82108/Desktop/2023-11-15-upbit-BTC-trade.csv")
