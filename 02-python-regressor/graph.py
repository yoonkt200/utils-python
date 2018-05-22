import numpy as np
import matplotlib.pyplot as plt

click_arr = np.arange(1000)
click_y = (click_arr + 80) * 10000 / 6
plt.plot(click_arr, click_y)


order_arr = np.arange(1000)
order_y = (1 + order_arr * 80) * 10000 / 6
plt.plot(order_arr, order_y)


price_arr = np.arange(10000, 100000, 100)
price_y = (1 + 80) * price_arr / 6
plt.plot(price_arr, price_y)


plt.plot(click_arr, click_y, 'r--', order_arr, order_y, 'bs', price_arr, price_y, 'g^')
plt.show()