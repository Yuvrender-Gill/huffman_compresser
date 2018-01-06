"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for value in text:
        if value in freq_dict:
            freq_dict[value] += 1
        else:
            freq_dict[value] = 1
    return freq_dict


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq1 = {1: 2}
    >>> t1 = huffman_tree(freq1)
    >>> sample_node = HuffmanNode(None, HuffmanNode(1))
    >>> t1 == sample_node
    True
    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    lst = sorted([(freq_dict[val], HuffmanNode(val)) for val in freq_dict])
    if len(lst) == 0:
        return HuffmanNode(None)   # Empty Dictionary Case
    elif len(lst) == 1:
        return HuffmanNode(None, lst[0][1])  # base case
    else:
        while len(lst) > 2:  # Tree with more Objects
            l, r = lst.pop(0), lst.pop(0)  # get two highest priority item
            lst.append((l[0] + r[0], HuffmanNode(None, l[1], r[1])))
            lst.sort()  # sort to optimize priority sequence
        return HuffmanNode(None, lst.pop(0)[1], lst.pop(0)[1])


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    def get_codes_helper(tree, l=''):
        """ Return a list of tuples mapping symbols from tree rooted at
        HuffmanNode to codes.

        @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
        @type l: str
        @rtype: dict(int,str)
        """
        if tree is None:  # if None get an empty list
            return []
        elif tree.is_leaf():  # if leaf get the symbol with the string
            return [(tree.symbol, l)]
        else:
            return get_codes_helper(tree.left, l + "0") + get_codes_helper(
                tree.right, l + "1")  # increment prefix l every time
    return dict(get_codes_helper(tree))  # call the helper function in dict


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(None, HuffmanNode(9), \
     HuffmanNode(11)), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number == 0
    True
    >>> tree.right.number == 2
    True
    >>> tree.number == 3
    True
    """
    def number_nodes_helper(tree, l):
        """ Assigns length of l to tree.number.

        @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
        @type l: list
        @rtype: NoneType
        """
        if tree is None:  # extra check for tree
            pass
        else:
            number_nodes_helper(tree.left, l)  # post order traversal left right
            number_nodes_helper(tree.right, l)
            if tree.symbol is None:
                tree.number = len(l)  # length of list starting from 0
                l.append(0)  # append the list to increase the node number
    helper = []
    number_nodes_helper(tree, helper)  # call the helper function


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    codes_dict = get_codes(tree)  # get the codes
    bits = sum([freq_dict[i] * len(codes_dict[i]) for i in freq_dict])
    symbols = sum([freq_dict[i] for i in freq_dict])  # total number of symbols
    return bits / symbols  # total bits divided with total symbols


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    str1 = "".join([codes[item] for item in text])  # make a string of all codes
    lst1 = [str1[i:i+8] for i in range(0, len(str1), 8)]  # slice at 8th step
    return bytes([bits_to_byte(item) for item in lst1])  # return in bytes


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    byte = bytes([])  # an empty bytes object
    if tree.left:
        byte += tree_to_bytes(tree.left)
        if tree.left.symbol is not None:  # if leaf return 0 and its symbol
            byte += bytes([0, tree.left.symbol])
        else:  # return 1 and its number
            byte += bytes([1, tree.left.number])
    if tree.right:
        byte += tree_to_bytes(tree.right)
        if tree.right.symbol is not None:  # if leaf return 0 and its symbol
            byte += bytes([0, tree.right.symbol])
        else:  # return 1 and its number
            byte += bytes([1, tree.right.number])  # add to bytes
    return byte  # return byte


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2) == HuffmanNode(None, HuffmanNode(None, \
    HuffmanNode(10, None, None), HuffmanNode(12, None, None)), \
    HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    True
    """
    node, tree = node_lst[root_index], HuffmanNode()
    if node.r_type == 0:  # assign data
        tree.right = HuffmanNode(node.r_data)
    else:  # Call the function recursively
        tree.right = generate_tree_general(node_lst, node.r_data)
    if node.l_type == 0:  # get the indexed node out
        tree.left = HuffmanNode(node.l_data)  # assign data
    else:  # Call the function recursively
        tree.left = generate_tree_general(node_lst, node.l_data)
    return tree


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2) == HuffmanNode(None, HuffmanNode(None, \
    HuffmanNode(5, None, None), HuffmanNode(7, None, None)), HuffmanNode(None, \
    HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    True
    """
    def postorder_helper(node_lst, root_index):  # helper function
        """ Return the tuple with root of the Huffman tree corresponding
            to node_lst[root_index].

            The function assumes that the list represents a tree in postorder.

            @param list[ReadNode] node_lst: a list of ReadNode objects
            @param int root_index: index in the node list
            @rtype: tuple of (HuffmanNode, int)
        """
        track_index, track_node = root_index, node_lst[root_index]
        tree = HuffmanNode()  # assign variables for node_lst and root_lst
        if track_node.r_type == 0:  # start traversing from right
            tree.right = HuffmanNode(track_node.r_data)  # if leaf
        else:
            right = postorder_helper(node_lst, track_index - 1)  # recurse down
            tree.right, track_index = right[0], right[1]  # modify right
            # and track new corresponding index
        if track_node.l_type == 0:  # start traversing left side
            tree.left = HuffmanNode(track_node.l_data)  # if leaf
        else:  # do recursion on left side now and modify tree and track index
            left = postorder_helper(node_lst, track_index - 1)
            tree.left, track_index = left[0], left[1]
        return tree, track_index  # return tuple
    return postorder_helper(node_lst, root_index)[0]  # get the tree out of
    # out of tuple


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    bit, symbols, cur_tree = ''.join([byte_to_bits(b) for b in text]), [], tree
    for i in bit:  # moving along the bit string
        if i == '0':  # if the first item in sting i 0 then move left
            cur_tree = cur_tree.left
        else:  # other wise go right down the tree
            cur_tree = cur_tree.right
        if cur_tree.symbol is not None:  # now reach a leaf
            symbols.append(cur_tree.symbol)  # append the leaf in the  symbols
            if len(symbols) == size:  # break the loop if size reached
                break  #
            cur_tree = tree  # assign the next tree
    return bytes(symbols)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, right, left)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    frequencies = sorted([(value, key) for key, value in freq_dict.items()],
                         reverse=True)
    queue = [tree]  # a list of tree to be used as queue API
    while len(queue) != 0:  # iterate over queue and get different nodes of tree
        t = queue.pop()  # get the tree out of queue
        if t.symbol:  # if leaf, change the value with first value in
            # frequencies which is according to increasing frequencies
            t.symbol = frequencies[0][1]  # assign the value
            frequencies.pop(0)  # pop out the last frequency
        else:  # if it is not a leaf then call recursively in post order
            queue.insert(0, t.left)
            queue.insert(0, t.right)

if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")

    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
