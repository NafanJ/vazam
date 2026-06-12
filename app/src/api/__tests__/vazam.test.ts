/**
 * vazam.test.ts — API client smoke tests (axios mocked, no network)
 */

import axios from 'axios';

jest.mock('axios', () => {
  const instance = {
    post: jest.fn(),
    get: jest.fn(),
    defaults: {baseURL: ''},
  };
  return {create: jest.fn(() => instance)};
});

import {identify, identifyShow, setBaseUrl} from '../vazam';

const client = (axios.create as jest.Mock).mock.results[0].value;

beforeEach(() => {
  client.post.mockReset();
  client.get.mockReset();
});

describe('identify', () => {
  it('posts multipart form to /identify', async () => {
    client.post.mockResolvedValue({data: {results: []}});

    const res = await identify({audioPath: 'file:///clip.wav', topK: 1});

    expect(client.post).toHaveBeenCalledTimes(1);
    const [url, form] = client.post.mock.calls[0];
    expect(url).toBe('/identify');
    expect(form).toBeInstanceOf(FormData);
    expect(res.results).toEqual([]);
  });
});

describe('identifyShow', () => {
  it('posts multipart form to /identify/show', async () => {
    client.post.mockResolvedValue({data: {show: null, speakers: {}}});

    const res = await identifyShow({audioPath: 'file:///clip.wav'});

    const [url, form] = client.post.mock.calls[0];
    expect(url).toBe('/identify/show');
    expect(form).toBeInstanceOf(FormData);
    expect(res.show).toBeNull();
    expect(res.speakers).toEqual({});
  });

  it('returns the inferred show when the backend finds consensus', async () => {
    const show = {
      show_id: 100,
      title: 'Cowboy Bebop',
      speakers_matched: 2,
      speakers_total: 3,
      score: 1.5,
    };
    client.post.mockResolvedValue({data: {show, speakers: {}}});

    const res = await identifyShow({audioPath: 'file:///clip.wav'});
    expect(res.show?.title).toBe('Cowboy Bebop');
    expect(res.show?.speakers_matched).toBe(2);
  });
});

describe('setBaseUrl', () => {
  it('strips a trailing slash', () => {
    setBaseUrl('http://10.0.0.5:8000/');
    expect(client.defaults.baseURL).toBe('http://10.0.0.5:8000');
  });
});
