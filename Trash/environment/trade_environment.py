import time
import numpy as np
import uuid
from collections import deque
from sortedcontainers import SortedDict


class Order:
    def __init__(self, order_id, agent_id, side, price, quantity):
        self.order_id = order_id
        self.agent_id = agent_id
        self.side = side  # 'buy' 或 'sell'
        self.price = price
        self.quantity = quantity
        self.timestamp = time.time()


class MatchingEngine:
    def __init__(self):
        # 使用 SortedDict 自動維持價格排序
        # asks (賣單): 價格由低到高排序。我們找賣單是從最便宜的開始找 (正向遍歷)
        self.asks = SortedDict()

        # bids (買單): 價格也是由低到高排序。但我們找買單是從出價最高的開始找 (反向遍歷)
        self.bids = SortedDict()

    def process_order(self, order):
        """處理單筆訂單，回傳這筆訂單產生的所有成交紀錄"""
        trades = []

        if order.side == "buy":
            trades = self._match_buy(order)
            if order.quantity > 0:
                self._add_to_book(self.bids, order)

        elif order.side == "sell":
            trades = self._match_sell(order)
            if order.quantity > 0:
                self._add_to_book(self.asks, order)

        return trades

    def _match_buy(self, buy_order):
        trades = []
        # ask_prices 已經是由低到高的迭代器
        # list() 這裡是為了避免在迭代過程中修改字典大小引發 RuntimeError
        ask_prices = list(self.asks.keys())

        for price in ask_prices:
            # 如果買方的出價小於市場最低賣價，或者買單已吃滿，則停止撮合
            if buy_order.price < price or buy_order.quantity == 0:
                break

            ask_queue = self.asks[price]
            while ask_queue and buy_order.quantity > 0:
                sell_order = ask_queue[0]  # 取出該價格最早掛單的賣方 (時間優先)

                traded_qty = min(buy_order.quantity, sell_order.quantity)
                trades.append(
                    {
                        "price": price,  # 成交價以掛單方 (Maker) 的價格為準
                        "qty": traded_qty,
                        "buyer_id": buy_order.agent_id,
                        "seller_id": sell_order.agent_id,
                        "timestamp": time.time(),
                    }
                )

                buy_order.quantity -= traded_qty
                sell_order.quantity -= traded_qty

                # 若該筆賣單被消耗完，將其移出佇列
                if sell_order.quantity == 0:
                    ask_queue.popleft()

            # 若該價格層位的賣單被清空，刪除該價格層
            if not ask_queue:
                del self.asks[price]

        return trades

    def _match_sell(self, sell_order):
        trades = []
        # 賣方要找願意出最高價的買方，因此使用 reversed() 反向遍歷
        bid_prices = list(reversed(self.bids.keys()))

        for price in bid_prices:
            # 如果賣方的要價大於市場最高買價，或者賣單已賣完，則停止撮合
            if sell_order.price > price or sell_order.quantity == 0:
                break

            bid_queue = self.bids[price]
            while bid_queue and sell_order.quantity > 0:
                buy_order = bid_queue[0]  # 取出該價格最早掛單的買方

                traded_qty = min(sell_order.quantity, buy_order.quantity)
                trades.append(
                    {
                        "price": price,  # 成交價以掛單方 (Maker) 的價格為準
                        "qty": traded_qty,
                        "buyer_id": buy_order.agent_id,
                        "seller_id": sell_order.agent_id,
                        "timestamp": time.time(),
                    }
                )

                sell_order.quantity -= traded_qty
                buy_order.quantity -= traded_qty

                if buy_order.quantity == 0:
                    bid_queue.popleft()

            if not bid_queue:
                del self.bids[price]

        return trades

    def _add_to_book(self, book, order):
        if order.price not in book:
            book[order.price] = deque()
        book[order.price].append(order)

    def get_lob_snapshot(self, depth=5):
        """
        深度功能：擷取訂單簿前 N 檔的狀態，作為 RL Agent 的 State
        回傳格式: { 'bids': [(price, volume), ...], 'asks': [(price, volume), ...] }
        """
        snapshot = {"bids": [], "asks": []}

        # 擷取最佳前 N 檔買單 (高到低)
        for price in reversed(self.bids.keys()):
            if len(snapshot["bids"]) >= depth:
                break
            volume = sum(order.quantity for order in self.bids[price])
            snapshot["bids"].append((price, volume))

        # 擷取最佳前 N 檔賣單 (低到高)
        for price in self.asks.keys():
            if len(snapshot["asks"]) >= depth:
                break
            volume = sum(order.quantity for order in self.asks[price])
            snapshot["asks"].append((price, volume))

        return snapshot


class MultiAgentMarketEnv:
    def __init__(
        self,
        num_agents,
        initial_cash=100000,
        initial_inventory=1000,
        heterogeneous_reward=False,
    ):
        self.engine = MatchingEngine()
        self.num_agents = num_agents
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.heterogeneous_reward = heterogeneous_reward

        # 記帳本：紀錄每個 agent 的現金與持倉量
        self.portfolios = {}
        # 為了計算 Reward，需要記錄前一步的資產總值
        self.prev_net_worth = {}

        # [新增] 儲存每個 Agent 完整的資產歷史（計算金融指標用）
        self.net_worth_history = {i: [] for i in range(num_agents)}

        # [新增] 市場全域新聞情緒分數值 [-1, 1]
        self.current_sentiment = 0.0

    def reset(self):
        """回合重置：清空訂單簿、重置所有 Agent 的資金與環境變數"""
        self.engine = MatchingEngine()
        self.current_sentiment = 0.0  # 重置情緒
        self.net_worth_history = {i: [] for i in range(self.num_agents)}

        for i in range(self.num_agents):
            self.portfolios[i] = {
                "cash": self.initial_cash,
                "inventory": self.initial_inventory,
            }
            initial_net_worth = self.initial_cash + (self.initial_inventory * 100.0)
            self.prev_net_worth[i] = initial_net_worth
            self.net_worth_history[i].append(initial_net_worth)

        initial_state = self.engine.get_lob_snapshot(depth=5)
        return {i: initial_state for i in range(self.num_agents)}

    def _simulate_noise_trader(self, mid_price):
        """[新增] 背景雜訊交易員：隨機向市場提供流動性，不計入 Agent 結算"""
        # 70% 機率每步注入隨機訂單，確保市場有對手盤
        if np.random.rand() > 0.3:
            side = "buy" if np.random.rand() > 0.5 else "sell"
            # 圍繞當前中間價隨機偏離 ±2% 掛單
            noise_price = round(mid_price * np.random.uniform(0.98, 1.02), 2)
            noise_qty = int(np.random.randint(5, 50))

            noise_order = Order(
                order_id=str(uuid.uuid4()),
                agent_id="NOISE_TRADER",
                side=side,
                price=noise_price,
                quantity=noise_qty,
            )
            # 雜訊單直接送入引擎撮合，若有剩餘就會留在訂單簿上提供流動性
            self.engine.process_order(noise_order)

    def step(self, actions):
        """
        核心互動迴圈
        actions: dict 格式，包含每個 agent 想下的單 {agent_id: act_vector}
        act_vector 為 4 維向量: [策略訊號, 方向訊號, 價格訊號, 數量訊號]，值域皆假設為 [-1, 1]
        """
        all_trades = []

        # 取得當下市場中間價，作為所有 Agent 掛單的基準參考
        mid_price = 100.0  # 預設基準
        if self.engine.bids and self.engine.asks:
            best_bid = list(self.engine.bids.keys())[-1]
            best_ask = list(self.engine.asks.keys())[0]
            mid_price = (best_bid + best_ask) / 2.0

        # 0. [新增] 模擬環境變化：更新新聞文本情緒分數 (隨機漫步模擬，並稍微加入趨勢延續性)
        sentiment_shock = np.random.normal(0, 0.1)
        self.current_sentiment = np.clip(
            self.current_sentiment * 0.9 + sentiment_shock, -1.0, 1.0
        )

        # 0. [新增] 引入背景雜訊交易員下單
        self._simulate_noise_trader(mid_price)

        # 1. 執行動作 (解析 4D 神經網路輸出並下單)
        for agent_id, act_vector in actions.items():
            strategy_signal = act_vector[0]
            side_signal = act_vector[1]
            price_signal = act_vector[2]
            qty_signal = act_vector[3]

            # (1) 判斷基礎買賣方向
            side = "buy" if side_signal > 0 else "sell"

            # (2) 計算數量比例 (將 [-1, 1] 映射到 [0, 1])
            trade_ratio = (qty_signal + 1) / 2.0

            order_price = mid_price
            order_quantity = 0

            # (3) 根據策略訊號決定價格與掛單模式
            if strategy_signal > 0.3:
                # 【造市商模式 Market Maker】
                price_offset = price_signal * 0.005
                order_price = round(mid_price * (1 + price_offset), 2)

            elif strategy_signal < -0.3:
                # 【動能交易模式 Trend Follower】
                if side == "buy":
                    order_price = round(mid_price * 1.1, 2)  # 溢價 10%
                else:
                    order_price = round(mid_price * 0.9, 2)  # 折價 10%

            else:
                # 【觀望模式 Hold】
                continue

            # (4) 根據實體資產限制，計算真實下單數量
            if side == "buy":
                available_cash = self.portfolios[agent_id]["cash"]
                if order_price > 0:
                    max_affordable_qty = int(available_cash / order_price)
                    order_quantity = int(max_affordable_qty * trade_ratio)
            elif side == "sell":
                available_inventory = self.portfolios[agent_id]["inventory"]
                order_quantity = int(available_inventory * trade_ratio)

            # (5) 生成並送出訂單
            if order_quantity > 0:
                order_id = str(uuid.uuid4())
                order = Order(order_id, agent_id, side, order_price, order_quantity)
                trades = self.engine.process_order(order)
                all_trades.extend(trades)

        # 2. 結算交割 (更新現金與庫存，避開 NOISE_TRADER)
        for trade in all_trades:
            buyer = trade["buyer_id"]
            seller = trade["seller_id"]
            trade_value = trade["price"] * trade["qty"]

            if buyer != "NOISE_TRADER":
                self.portfolios[buyer]["cash"] -= trade_value
                self.portfolios[buyer]["inventory"] += trade["qty"]

            if seller != "NOISE_TRADER":
                self.portfolios[seller]["cash"] += trade_value
                self.portfolios[seller]["inventory"] -= trade["qty"]

        # 3. 取得新狀態 (Next State)
        next_state = self.engine.get_lob_snapshot(depth=5)

        # 重新計算撮合後的新中間價，用於準確評估資產
        if self.engine.bids and self.engine.asks:
            best_bid = list(self.engine.bids.keys())[-1]
            best_ask = list(self.engine.asks.keys())[0]
            mid_price = (best_bid + best_ask) / 2.0

        # 4. 計算獎勵 (Reward = 資產變化量)
        rewards = {}
        for i in range(self.num_agents):
            # 基礎的總資產 (PnL)
            current_net_worth = self.portfolios[i]["cash"] + (
                self.portfolios[i]["inventory"] * mid_price
            )
            delta_pnl = current_net_worth - self.prev_net_worth[i]

            # [新增] 將當前淨資產存入歷史紀錄
            self.net_worth_history[i].append(current_net_worth)

            # --- 根據 Flag 切換 Reward 模式 ---
            if not self.heterogeneous_reward:
                # 【模式 A：純淨市場】
                rewards[i] = delta_pnl

            else:
                # 【模式 B：異質代理人市場 (Reward Shaping)】
                if i == 0:
                    # Agent 0：造市商 (Market Maker) - 庫存懲罰
                    inventory_penalty = abs(self.portfolios[i]["inventory"]) * 0.1
                    rewards[i] = delta_pnl - inventory_penalty

                elif i == 1:
                    # Agent 1：保守型交易員 (Conservative) - 手續費/過度交易懲罰
                    trade_penalty = 0
                    inventory_change = abs(current_net_worth - self.prev_net_worth[i])
                    if inventory_change > 0:
                        trade_penalty = 5.0
                    rewards[i] = delta_pnl - trade_penalty

                elif i == 2:
                    # Agent 2：嗜血動能投機客 (Aggressive) - 原汁原味的 PnL
                    rewards[i] = delta_pnl

                else:
                    rewards[i] = delta_pnl

            # 記錄當前資產供下回合使用
            self.prev_net_worth[i] = current_net_worth

        # 5. 回傳 RL 標準格式
        dones = {i: False for i in range(self.num_agents)}
        infos = {i: self.portfolios[i] for i in range(self.num_agents)}

        return {i: next_state for i in range(self.num_agents)}, rewards, dones, infos


def flatten_state(state_dict, portfolio, sentiment_score, depth=5):
    """
    將環境輸出的 state 轉換為神經網路可吃的 1D numpy array
    [修改] 引入文本情緒分數，神經網路輸入維度從 22 提升至 23
    輸出維度: 買價量(10) + 賣價量(10) + 現金(1) + 庫存(1) + 情緒分數(1) = 23
    """
    features = []

    # 處理買單 (Bids)
    for i in range(depth):
        if i < len(state_dict["bids"]):
            features.append(float(state_dict["bids"][i][0]))  # Price
            features.append(float(state_dict["bids"][i][1]))  # Quantity
        else:
            features.extend([0.0, 0.0])  # 若無掛單則補 0

    # 處理賣單 (Asks)
    for i in range(depth):
        if i < len(state_dict["asks"]):
            features.append(float(state_dict["asks"][i][0]))  # Price
            features.append(float(state_dict["asks"][i][1]))  # Quantity
        else:
            features.extend([0.0, 0.0])

    # 加入自身資產狀態
    features.append(float(portfolio["cash"]))
    features.append(float(portfolio["inventory"]))

    # [新增] 加入外部大環境變數 (LLM 情緒分數)
    features.append(float(sentiment_score))

    return np.array(features, dtype=np.float32)


def calculate_financial_metrics(net_worth_history):
    """
    [新增] 計算專業量化交易指標 (夏普比率與最大回撤)
    """
    net_worths = np.array(net_worth_history)

    # 計算每一步的收益率 (Returns)
    returns = np.diff(net_worths) / net_worths[:-1]
    returns = np.nan_to_num(returns)

    # 計算夏普比率 (Sharpe Ratio)
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0

    # 計算最大回撤 (Maximum Drawdown)
    peaks = np.maximum.accumulate(net_worths)
    drawdowns = (net_worths - peaks) / peaks
    max_drawdown = np.min(drawdowns)

    return sharpe_ratio, max_drawdown
