{
    "model_path": {
    		"news": "../data/train.json",
    		"user": "../data/valid.json"
    },
    "pool_path": {
    		"news_vec": "../data/train.json",
    		"user_vec": "../data/valid.json",
    		"candidates": "../data/test.json"
    },
    "item_col": { 
    		"news_id":"page.item.id",
				"user_id":"cookie_mapping.et_token",
				"event_time":"event_timestamp"
    }
}